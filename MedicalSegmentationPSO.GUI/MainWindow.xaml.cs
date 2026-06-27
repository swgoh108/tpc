using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using MedicalSegmentationPSO.Benchmark;
using Microsoft.Win32;
using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;
using Window = System.Windows.Window;

namespace MedicalSegmentationPSO.GUI
{
	public class AlgoCompareResult
	{
		public string Algorithm { get; set; } = "";
		public double TimeMs { get; set; }
		public double Speedup { get; set; }
		public double Efficiency { get; set; }
		public bool IsBest { get; set; }
	}

	public partial class MainWindow : Window
	{
		private byte[]? currentPixels;
		private int imgWidth, imgHeight;
		private double sequentialBaseline = 0;
		private BitmapSource? lastSegmentedImage;

		private CNNClassifier cnnClassifier = new CNNClassifier();
		private bool cnnModelLoaded = false;

		public MainWindow()
		{
			InitializeComponent();
			txtCores.Text = $"CPU Cores / Threads: {Environment.ProcessorCount}";

			string[] modelCandidates =
			{
				@"C:\Users\User\source\repos\tpc\BrainTumorCNN.pt",
				Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "BrainTumorCNN.pt"),
				Path.Combine(Directory.GetCurrentDirectory(),      "BrainTumorCNN.pt"),
			};

			foreach (string candidate in modelCandidates)
			{
				if (File.Exists(candidate))
				{
					TryLoadCnnModel(candidate);
					break;
				}
			}
		}

		private void Close_Click(object sender, RoutedEventArgs e) => Close();
		private void Minimize_Click(object sender, RoutedEventArgs e)
			=> WindowState = WindowState.Minimized;
		private void Window_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
		{
			if (e.LeftButton == MouseButtonState.Pressed) DragMove();
		}

		private void ShowProgress(string message)
		{
			txtProgress.Text = message;
			pnlProgress.Visibility = Visibility.Visible;
		}

		private void HideProgress()
		{
			pnlProgress.Visibility = Visibility.Collapsed;
		}

		private void BtnBrowse_Click(object sender, RoutedEventArgs e)
		{
			var dlg = new OpenFileDialog
			{
				Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp",
				Title = "Select MRI / CT Image"
			};
			if (dlg.ShowDialog() != true) return;

			currentPixels = ImageProcessor.LoadGrayscalePixels(
				dlg.FileName, out imgWidth, out imgHeight);

			using var mat = Cv2.ImRead(dlg.FileName, ImreadModes.Grayscale);
			imgOriginal.Source = MatToBitmapSource(mat);
			imgSegmented.Source = null;
			lastSegmentedImage = null;
			sequentialBaseline = 0;

			pnlComparison.Visibility = Visibility.Collapsed;

			btnClassify.IsEnabled = true;
			txtClassification.Text = "Prediction: —";
			txtClassNote.Text = cnnModelLoaded
				? "Run PSO to segment, then click Classify."
				: "⚠ No trained model found — results may be inaccurate. Run PSO first, then Classify.";

			SetStatus("Image loaded. Select a PSO algorithm or click Run All.", "", "", "", "");
		}

		private void BtnSequential_Click(object sender, RoutedEventArgs e)
			=> RunSegmentation(new SequentialPSOAdapter(), "Sequential PSO");

		private void BtnMultiCore_Click(object sender, RoutedEventArgs e)
			=> RunSegmentation(new MultiCorePSOAdapter(), "Multi-Core PSO");

		private void BtnPLINQ_Click(object sender, RoutedEventArgs e)
			=> RunSegmentation(new PLINQPSOAdapter(), "PLINQ PSO");

		private void BtnShared_Click(object sender, RoutedEventArgs e)
			=> RunSegmentation(new SharedMemoryPSOAdapter(), "Shared Memory PSO");

		private void BtnGPU_Click(object sender, RoutedEventArgs e)
			=> RunSegmentation(new GPUPSOAdapter(), "GPU Accel PSO");

		private async void RunSegmentation(IPSOAlgorithm algorithm, string name)
		{
			if (currentPixels == null)
			{
				MessageBox.Show("Please load an image first!");
				return;
			}

			SetAllButtonsEnabled(false);
			ShowProgress($"Running {name} — please wait…");
			txtStatus.Text = $"Running {name}…";

			bool[] brainMask = BuildBrainMask(currentPixels, imgWidth, imgHeight);
			byte[] brainPixels = MaskPixels(currentPixels, brainMask);

			var sw = Stopwatch.StartNew();
			double[] thresholds = await Task.Run(() => algorithm.Run(brainPixels));
			sw.Stop();

			HideProgress();

			DisplaySegmentation(thresholds, brainMask);

			double timeMs = sw.Elapsed.TotalMilliseconds;
			if (name.Contains("Sequential")) sequentialBaseline = timeMs;

			double speedup = sequentialBaseline > 0 ? sequentialBaseline / timeMs : 1.0;
			double efficiency = speedup / Environment.ProcessorCount;

			int tumorPx = CountTumorPixels(currentPixels, thresholds, brainMask);
			double tumorPct = tumorPx * 100.0 / currentPixels.Length;

			SetStatus(
				$"{name} completed.",
				$"Execution Time : {timeMs:F2} ms",
				$"Speedup        : {speedup:F2}×  (vs Sequential baseline)",
				$"Efficiency     : {efficiency:F4}",
				$"Tumor Region   : {tumorPx:N0} px ({tumorPct:F1}%)  |  " +
				$"Thresholds: {thresholds[0]:F1} / {thresholds[1]:F1} / {thresholds[2]:F1}"
			);

			SetAllButtonsEnabled(true);
		}

		private async void BtnRunAll_Click(object sender, RoutedEventArgs e)
		{
			if (currentPixels == null)
			{
				MessageBox.Show("Please load an image first!");
				return;
			}

			SetAllButtonsEnabled(false);
			pnlComparison.Visibility = Visibility.Collapsed;

			var algorithms = new (IPSOAlgorithm Algo, string Name)[]
			{
				(new SequentialPSOAdapter(),   "Sequential PSO"),
				(new MultiCorePSOAdapter(),    "Multi-Core PSO"),
				(new PLINQPSOAdapter(),        "PLINQ PSO"),
				(new SharedMemoryPSOAdapter(), "Shared Memory PSO"),
				(new GPUPSOAdapter(),          "GPU Accel PSO"),
			};

			bool[] brainMask = BuildBrainMask(currentPixels, imgWidth, imgHeight);
			byte[] brainPixels = MaskPixels(currentPixels, brainMask);
			double seqTime = 0;
			double[] lastThresholds = null!;
			var compareResults = new List<AlgoCompareResult>();

			foreach (var (algo, name) in algorithms)
			{
				ShowProgress($"Running {name} ({algorithms.ToList().IndexOf((algo, name)) + 1} / {algorithms.Length})…");
				txtStatus.Text = $"Running {name}…";

				var sw = Stopwatch.StartNew();
				double[] thresholds = await Task.Run(() => algo.Run(brainPixels));
				sw.Stop();
				double ms = sw.Elapsed.TotalMilliseconds;

				if (name.Contains("Sequential")) seqTime = ms;

				double speedup = seqTime > 0 ? seqTime / ms : 1.0;
				double efficiency = speedup / Environment.ProcessorCount;

				compareResults.Add(new AlgoCompareResult
				{
					Algorithm = name,
					TimeMs = ms,
					Speedup = speedup,
					Efficiency = efficiency,
				});

				lastThresholds = thresholds;
			}

			HideProgress();

			var fastest = compareResults.Skip(1).OrderBy(r => r.TimeMs).FirstOrDefault();
			if (fastest != null) fastest.IsBest = true;

			sequentialBaseline = seqTime;

			DisplaySegmentation(lastThresholds, brainMask);

			RenderComparisonTable(compareResults);

			int tumorPx = CountTumorPixels(currentPixels, lastThresholds, brainMask);
			double tumorPct = tumorPx * 100.0 / currentPixels.Length;

			SetStatus(
				"All 5 algorithms completed. See comparison below.",
				$"Sequential Baseline       : {seqTime:F2} ms",
				$"Best Parallel Speedup     : {compareResults.Skip(1).Max(r => r.Speedup):F2}×",
				$"Cores Available           : {Environment.ProcessorCount}",
				$"Tumor Region (last run)   : {tumorPx:N0} px ({tumorPct:F1}%)"
			);

			SetAllButtonsEnabled(true);
		}

		private void RenderComparisonTable(List<AlgoCompareResult> results)
		{
			spComparisonRows.Children.Clear();

			foreach (var r in results)
			{
				var row = new Grid { Margin = new Thickness(0, 3, 0, 3) };
				row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
				row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(130) });
				row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(100) });
				row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(110) });

				if (r.IsBest)
					row.Background = new SolidColorBrush(
						System.Windows.Media.Color.FromArgb(50, 16, 185, 129));

				Color nameColor = r.Algorithm.Contains("Sequential")
					? System.Windows.Media.Color.FromRgb(148, 163, 184)
					: System.Windows.Media.Color.FromRgb(255, 255, 255);

				Color speedColor = r.Speedup >= 2.0
					? System.Windows.Media.Color.FromRgb(52, 211, 153)
					: r.Speedup >= 1.0
						? System.Windows.Media.Color.FromRgb(251, 191, 36)
						: System.Windows.Media.Color.FromRgb(248, 113, 113);

				string bestTag = r.IsBest ? "  ★ fastest" : "";

				AddCell(row, 0, r.Algorithm + bestTag, new SolidColorBrush(nameColor), TextAlignment.Left);
				AddCell(row, 1, $"{r.TimeMs:F2} ms", Brushes.LightBlue, TextAlignment.Right);
				AddCell(row, 2, $"{r.Speedup:F2}×", new SolidColorBrush(speedColor), TextAlignment.Right);
				AddCell(row, 3, $"{r.Efficiency:F4}", Brushes.LightGreen, TextAlignment.Right);

				spComparisonRows.Children.Add(row);
			}

			int cores = Environment.ProcessorCount;
			double maxSpdup = results.Skip(1).Max(r => r.Speedup);

			txtComparisonNote.Text =
				$"Cores used: {cores}  |  " +
				$"Best speedup: {maxSpdup:F2}×  |  " +
				$"Theoretical max (Amdahl): {cores}×  |  " +
				$"Efficiency = Speedup ÷ Cores";

			pnlComparison.Visibility = Visibility.Visible;
		}

		private static void AddCell(Grid grid, int col, string text,
									Brush brush, TextAlignment align)
		{
			var tb = new TextBlock
			{
				Text = text,
				Foreground = brush,
				FontSize = 13,
				TextAlignment = align,
				VerticalAlignment = VerticalAlignment.Center,
			};
			Grid.SetColumn(tb, col);
			grid.Children.Add(tb);
		}

		private async void BtnBenchmark3_Click(object sender, RoutedEventArgs e)
		{
			if (currentPixels == null) { MessageBox.Show("Please load an image first!"); return; }

			const int RUNS = 3;

			var algorithms = new (IPSOAlgorithm Algo, string Name)[]
			{
				(new SequentialPSOAdapter(),   "Sequential PSO"),
				(new MultiCorePSOAdapter(),    "Multi-Core PSO"),
				(new PLINQPSOAdapter(),        "PLINQ PSO"),
				(new SharedMemoryPSOAdapter(), "Shared Memory PSO"),
				(new GPUPSOAdapter(),          "GPU Accel PSO"),
			};

			bool[] brainMask = BuildBrainMask(currentPixels, imgWidth, imgHeight);
			byte[] brainPixels = MaskPixels(currentPixels, brainMask);

			SetAllButtonsEnabled(false);

			var lines = new List<string>();
			double seqAvg = 0;

			for (int algoIdx = 0; algoIdx < algorithms.Length; algoIdx++)
			{
				var (algo, name) = algorithms[algoIdx];

				ShowProgress($"Benchmarking {name} — run 1 of {RUNS}…");
				txtStatus.Text = $"Benchmarking {name} ({RUNS} runs)…";

				var times = new double[RUNS];
				for (int i = 0; i < RUNS; i++)
				{
					txtProgress.Text = $"Benchmarking {name} — run {i + 1} of {RUNS}…";
					var sw = Stopwatch.StartNew();
					await Task.Run(() => algo.Run(brainPixels));
					sw.Stop();
					times[i] = sw.Elapsed.TotalMilliseconds;
				}

				double avg = times.Average();
				double stddev = Math.Sqrt(times.Average(t => Math.Pow(t - avg, 2)));

				if (name.Contains("Sequential")) seqAvg = avg;

				double speedup = seqAvg > 0 ? seqAvg / avg : 1.0;
				double efficiency = speedup / Environment.ProcessorCount;

				lines.Add($"[{name}]");
				for (int i = 0; i < RUNS; i++)
					lines.Add($"  Run {i + 1}: {times[i]:F2} ms");
				lines.Add($"  Avg: {avg:F2} ms  |  StdDev: {stddev:F2}  |  Speedup: {speedup:F2}×  |  Efficiency: {efficiency:F4}");
				lines.Add("");
			}

			HideProgress();
			SetAllButtonsEnabled(true);

			string report = string.Join("\n", lines);
			txtStatus.Text = "Benchmark completed (3 runs × 5 algorithms).";
			txtAvgTime.Text = "Benchmark done — see results below.";

			MessageBox.Show(report,
				"Benchmark Results (3 Runs × 5 Algorithms)",
				MessageBoxButton.OK,
				MessageBoxImage.Information);
		}


		private void TryLoadCnnModel(string path)
		{
			try
			{
				cnnClassifier.Load(path);
				cnnModelLoaded = true;
			}
			catch
			{
				cnnModelLoaded = false;
			}
		}

		private async void BtnClassify_Click(object sender, RoutedEventArgs e)
		{
			if (currentPixels == null) { MessageBox.Show("Load an image first."); return; }

			SetAllButtonsEnabled(false);
			ShowProgress("Running CNN classification…");
			txtClassification.Text = "Classifying…";

			try
			{
				const int SIZE = 224;

				using var src = new Mat(imgHeight, imgWidth, MatType.CV_8UC1);
				for (int y = 0; y < imgHeight; y++)
					for (int x = 0; x < imgWidth; x++)
						src.Set(y, x, currentPixels[y * imgWidth + x]);

				using var resized = new Mat();
				Cv2.Resize(src, resized, new OpenCvSharp.Size(SIZE, SIZE));

				byte[] buf = new byte[SIZE * SIZE];
				Marshal.Copy(resized.Data, buf, 0, buf.Length);

				float[] data = new float[buf.Length];
				for (int i = 0; i < buf.Length; i++) data[i] = buf[i] / 255f;

				var tensor = torch.tensor(data, dtype: ScalarType.Float32)
								  .reshape(1, 1, SIZE, SIZE);

				int predIdx = await Task.Run(() => cnnClassifier.Predict(tensor));

				string label = predIdx >= 0 && predIdx < TumorLabels.Classes.Length
					? TumorLabels.Classes[predIdx]
					: $"Class {predIdx}";

				string display = label switch
				{
					"glioma" => "🔴 Glioma Detected",
					"meningioma" => "🟠 Meningioma Detected",
					"pituitary" => "🟡 Pituitary Tumor Detected",
					"notumor" => "🟢 No Tumor Detected",
					_ => label
				};

				txtClassification.Text = $"Prediction: {display}";

				string modelNote = cnnModelLoaded
					? "Trained model (BrainTumorCNN.pt)"
					: "⚠ Untrained model — load BrainTumorCNN.pt for accurate results";

				txtClassNote.Text =
					$"{modelNote}  |  Raw label: {label}  |  " +
					$"Classes: {string.Join(", ", TumorLabels.Classes)}";
			}
			catch (Exception ex)
			{
				txtClassification.Text = "Prediction: Error";
				MessageBox.Show($"Classification error:\n{ex.Message}",
					"Error", MessageBoxButton.OK, MessageBoxImage.Error);
			}
			finally
			{
				HideProgress();
				SetAllButtonsEnabled(true);
			}
		}

		private void BtnSaveSegmented_Click(object sender, RoutedEventArgs e)
		{
			if (lastSegmentedImage == null)
			{
				MessageBox.Show("No segmented image to save. Run a PSO algorithm first.");
				return;
			}

			var dlg = new SaveFileDialog
			{
				Filter = "PNG Image|*.png|JPEG Image|*.jpg",
				FileName = "segmented_output.png"
			};
			if (dlg.ShowDialog() != true) return;

			BitmapEncoder encoder = dlg.FilterIndex == 2
				? new JpegBitmapEncoder()
				: new PngBitmapEncoder();

			encoder.Frames.Add(BitmapFrame.Create(lastSegmentedImage));
			using var stream = File.Create(dlg.FileName);
			encoder.Save(stream);

			MessageBox.Show("Segmented image saved successfully!");
		}


		private void DisplaySegmentation(double[] thresholds, bool[] brainMask)
		{
			byte[] classMap = BuildClassMap(currentPixels!, thresholds);
			bool[] tumorMask = ExtractTumorMask(classMap, brainMask, imgWidth, imgHeight);
			lastSegmentedImage = CreateTumorOverlay(currentPixels!, tumorMask, imgWidth, imgHeight);
			imgSegmented.Source = lastSegmentedImage;
		}

		private int CountTumorPixels(byte[] pixels, double[] thresholds, bool[] brainMask)
		{
			byte[] classMap = BuildClassMap(pixels, thresholds);
			bool[] tumorMask = ExtractTumorMask(classMap, brainMask, imgWidth, imgHeight);
			return tumorMask.Count(b => b);
		}

		private bool[] BuildBrainMask(byte[] pixels, int w, int h)
		{
			const byte bgThreshold = 20;

			bool[] isBg = new bool[pixels.Length];
			var queue = new Queue<int>();

			void Seed(int x, int y)
			{
				int i = y * w + x;
				if (pixels[i] <= bgThreshold && !isBg[i])
				{
					isBg[i] = true;
					queue.Enqueue(i);
				}
			}

			for (int x = 0; x < w; x++) { Seed(x, 0); Seed(x, h - 1); }
			for (int y = 0; y < h; y++) { Seed(0, y); Seed(w - 1, y); }

			int[] dx4 = { -1, 1, 0, 0 };
			int[] dy4 = { 0, 0, -1, 1 };

			while (queue.Count > 0)
			{
				int cur = queue.Dequeue();
				int cy = cur / w, cx = cur % w;
				for (int d = 0; d < 4; d++)
				{
					int nx = cx + dx4[d], ny = cy + dy4[d];
					if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
					int ni = ny * w + nx;
					if (!isBg[ni] && pixels[ni] <= bgThreshold)
					{
						isBg[ni] = true;
						queue.Enqueue(ni);
					}
				}
			}

			bool[] head = new bool[pixels.Length];
			for (int i = 0; i < pixels.Length; i++)
				head[i] = !isBg[i] && pixels[i] > bgThreshold;

			int erodeRadius = Math.Max(3, (int)(Math.Min(w, h) * 0.06));
			bool[] eroded = ErodeCircular(head, w, h, erodeRadius);

			bool[] largest = LargestComponent(eroded, w, h);

			int dilateRadius = Math.Max(2, (int)(Math.Min(w, h) * 0.04));
			bool[] dilated = DilateCircular(largest, w, h, dilateRadius);

			return dilated;
		}

		private bool[] DilateCircular(bool[] mask, int w, int h, int radius)
		{
			bool[] result = new bool[mask.Length];

			var offsets = new List<(int dx, int dy)>();
			for (int dy = -radius; dy <= radius; dy++)
				for (int dx = -radius; dx <= radius; dx++)
					if (dx * dx + dy * dy <= radius * radius)
						offsets.Add((dx, dy));

			for (int y = 0; y < h; y++)
			{
				for (int x = 0; x < w; x++)
				{
					if (mask[y * w + x])
					{
						result[y * w + x] = true;
						continue;
					}

					foreach (var (dx, dy) in offsets)
					{
						int nx = x + dx;
						int ny = y + dy;
						if (nx >= 0 && nx < w && ny >= 0 && ny < h && mask[ny * w + nx])
						{
							result[y * w + x] = true;
							break;
						}
					}
				}
			}

			return result;
		}

		private bool[] ErodeCircular(bool[] mask, int w, int h, int radius)
		{
			bool[] result = new bool[mask.Length];
			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
				{
					if (!mask[y * w + x]) continue;
					bool fits = true;
					for (int dy = -radius; dy <= radius && fits; dy++)
						for (int dx = -radius; dx <= radius && fits; dx++)
						{
							if (dx * dx + dy * dy > radius * radius) continue;
							int nx = x + dx, ny = y + dy;
							if (nx < 0 || nx >= w || ny < 0 || ny >= h || !mask[ny * w + nx])
								fits = false;
						}
					result[y * w + x] = fits;
				}
			return result;
		}

		private bool[] LargestComponent(bool[] mask, int w, int h)
		{
			int[] labels = new int[mask.Length];
			Array.Fill(labels, -1);
			int next = 0;
			var sizes = new List<int>();
			int[] dx4 = { -1, 1, 0, 0 };
			int[] dy4 = { 0, 0, -1, 1 };

			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
				{
					int idx = y * w + x;
					if (!mask[idx] || labels[idx] >= 0) continue;
					var q = new Queue<int>(); q.Enqueue(idx);
					labels[idx] = next; int sz = 0;
					while (q.Count > 0)
					{
						int cur = q.Dequeue(); sz++;
						int cy = cur / w, cx = cur % w;
						for (int d = 0; d < 4; d++)
						{
							int nx = cx + dx4[d], ny = cy + dy4[d];
							if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
							int ni = ny * w + nx;
							if (mask[ni] && labels[ni] < 0) { labels[ni] = next; q.Enqueue(ni); }
						}
					}
					sizes.Add(sz); next++;
				}

			if (next == 0) return mask;
			int maxLabel = sizes.IndexOf(sizes.Max());
			bool[] result = new bool[mask.Length];
			for (int i = 0; i < mask.Length; i++)
				result[i] = labels[i] == maxLabel;
			return result;
		}

		private byte[] MaskPixels(byte[] pixels, bool[] mask)
		{
			byte[] out_ = new byte[pixels.Length];
			for (int i = 0; i < pixels.Length; i++)
				out_[i] = mask[i] ? pixels[i] : (byte)0;
			return out_;
		}

		private byte[] BuildClassMap(byte[] pixels, double[] thresholds)
		{
			double[] T = (double[])thresholds.Clone(); Array.Sort(T);
			byte[] map = new byte[pixels.Length];
			for (int i = 0; i < pixels.Length; i++)
				map[i] = pixels[i] <= T[0] ? (byte)0 :
						  pixels[i] <= T[1] ? (byte)1 :
						  pixels[i] <= T[2] ? (byte)2 : (byte)3;
			return map;
		}

		private bool[] ExtractTumorMask(byte[] classMap, bool[] brainMask, int w, int h)
		{
			bool[] candidate = new bool[classMap.Length];
			for (int i = 0; i < classMap.Length; i++)
				candidate[i] = classMap[i] == 3 && brainMask[i];

			if (candidate.Count(b => b) < w * h * 0.001)
				for (int i = 0; i < classMap.Length; i++)
					candidate[i] = (classMap[i] == 3 || classMap[i] == 2) && brainMask[i];

			bool[] result = LargestComponent(candidate, w, h);
			int count = result.Count(b => b);
			double pct = count * 100.0 / (w * h);

			if (pct > 40 || count < 10) return new bool[classMap.Length];
			return result;
		}

		private BitmapSource CreateTumorOverlay(byte[] original, bool[] tumorMask,
												 int width, int height)
		{
			byte[] out_ = new byte[width * height * 4];
			for (int i = 0; i < original.Length; i++)
			{
				int idx = i * 4;
				byte gray = original[i];
				if (tumorMask[i])
				{
					out_[idx + 0] = 30;
					out_[idx + 1] = 30;
					out_[idx + 2] = 255;
					out_[idx + 3] = 255;
				}
				else
				{
					byte dim = (byte)(gray * 0.55);
					out_[idx + 0] = dim;
					out_[idx + 1] = dim;
					out_[idx + 2] = dim;
					out_[idx + 3] = 255;
				}
			}

			var wb = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
			wb.WritePixels(new Int32Rect(0, 0, width, height), out_, width * 4, 0);
			wb.Freeze();
			return wb;
		}

		private BitmapSource MatToBitmapSource(Mat mat)
		{
			int w = mat.Width, h = mat.Height;
			byte[] pix = new byte[w * h];
			mat.GetArray(out pix);
			var wb = new WriteableBitmap(w, h, 96, 96, PixelFormats.Gray8, null);
			wb.WritePixels(new Int32Rect(0, 0, w, h), pix, w, 0);
			wb.Freeze();
			return wb;
		}

		private void SetAllButtonsEnabled(bool enabled)
		{
			btnClassify.IsEnabled = enabled && (currentPixels != null);
			btnRunAll.IsEnabled = enabled;
			foreach (var btn in FindVisualChildren<Button>(this))
			{
				if (btn == btnClassify) continue;
				btn.IsEnabled = enabled;
			}
		}

		private static IEnumerable<T> FindVisualChildren<T>(DependencyObject parent)
			where T : DependencyObject
		{
			int count = System.Windows.Media.VisualTreeHelper.GetChildrenCount(parent);
			for (int i = 0; i < count; i++)
			{
				var child = System.Windows.Media.VisualTreeHelper.GetChild(parent, i);
				if (child is T t) yield return t;
				foreach (var sub in FindVisualChildren<T>(child)) yield return sub;
			}
		}

		private void SetStatus(string status, string time, string speedup,
							   string efficiency, string tumor)
		{
			txtStatus.Text = status;
			txtTime.Text = time;
			txtAvgTime.Text = "";
			txtSpeedup.Text = speedup;
			txtEfficiency.Text = efficiency;
			txtTumor.Text = tumor;
		}
	}
}