using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using MedicalSegmentationPSO.Benchmark;
using Microsoft.Win32;
using OpenCvSharp;
using Window = System.Windows.Window;

namespace MedicalSegmentationPSO.GUI
{
	public partial class MainWindow : Window
	{
		private byte[]? currentPixels;
		private int imgWidth, imgHeight;
		private double sequentialBaseline = 0;
		private BitmapSource? lastSegmentedImage;

		public MainWindow()
		{
			InitializeComponent();
			txtCores.Text = $"CPU Cores: {Environment.ProcessorCount}";
		}

		private void Close_Click(object sender, RoutedEventArgs e) => Close();

		private void Window_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
		{
			if (e.LeftButton == MouseButtonState.Pressed)
				DragMove();
		}

		private void Minimize_Click(object sender, RoutedEventArgs e)
			=> WindowState = WindowState.Minimized;

		private void MaxRestore_Click(object sender, RoutedEventArgs e)
			=> WindowState = WindowState == WindowState.Maximized
				? WindowState.Normal
				: WindowState.Maximized;

		private void BtnBrowse_Click(object sender, RoutedEventArgs e)
		{
			var dlg = new OpenFileDialog
			{
				Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp"
			};

			if (dlg.ShowDialog() == true)
			{
				currentPixels = ImageProcessor.LoadGrayscalePixels(
					dlg.FileName,
					out imgWidth,
					out imgHeight
				);

				using var mat = Cv2.ImRead(dlg.FileName, ImreadModes.Grayscale);
				imgOriginal.Source = MatToBitmapSource(mat);

				imgSegmented.Source = null;
				txtStatus.Text = "Image loaded. Select a PSO algorithm.";
				txtTime.Text = "";
				txtAvgTime.Text = "";
				txtSpeedup.Text = "";
				txtEfficiency.Text = "";
				txtTumor.Text = "";
				sequentialBaseline = 0;
			}
		}

		private async void RunSegmentation(IPSOAlgorithm algorithm, string name)
		{
			if (currentPixels == null)
			{
				MessageBox.Show("Please load an image first!");
				return;
			}

			txtStatus.Text = $"Running {name}...";

			bool[] brainMask = BuildBrainMask(currentPixels, imgWidth, imgHeight);

			byte[] brainPixels = MaskPixels(currentPixels, brainMask);

			var sw = Stopwatch.StartNew();

			double[] thresholds = await System.Threading.Tasks.Task.Run(() =>
				algorithm.Run(brainPixels)
			);

			sw.Stop();

			byte[] classMap = BuildClassMap(currentPixels, thresholds);

			bool[] tumorMask = ExtractTumorMask(classMap, brainMask, imgWidth, imgHeight);

			lastSegmentedImage = CreateTumorOverlay(currentPixels, tumorMask, imgWidth, imgHeight);
			imgSegmented.Source = lastSegmentedImage;

			double timeMs = sw.Elapsed.TotalMilliseconds;

			if (name.Contains("Sequential"))
				sequentialBaseline = timeMs;

			double speedup = sequentialBaseline > 0 ? sequentialBaseline / timeMs : 1.0;
			double efficiency = speedup / Environment.ProcessorCount;

			int tumorPx = tumorMask.Count(b => b);
			double tumorPct = tumorPx * 100.0 / currentPixels.Length;

			txtTime.Text = $"Execution Time: {timeMs:F2} ms";
			txtAvgTime.Text = $"Avg Time: {timeMs:F2} ms";
			txtSpeedup.Text = $"Speedup: {speedup:F2}x";
			txtEfficiency.Text = $"Efficiency: {efficiency:F4}";
			txtTumor.Text = $"Tumor Region: {tumorPx:N0} px ({tumorPct:F1}%)  |  " +
								$"Thresholds: {thresholds[0]:F1} / {thresholds[1]:F1} / {thresholds[2]:F1}";
			txtStatus.Text = $"{name} Completed Successfully!";
		}

		private bool[] BuildBrainMask(byte[] pixels, int w, int h)
		{
			const byte bgThreshold = 20;

			bool[] isBg = new bool[pixels.Length];
			var queue = new Queue<int>();

			void SeedCorner(int x, int y)
			{
				int idx = y * w + x;
				if (pixels[idx] <= bgThreshold && !isBg[idx])
				{
					isBg[idx] = true;
					queue.Enqueue(idx);
				}
			}

			for (int x = 0; x < w; x++) { SeedCorner(x, 0); SeedCorner(x, h - 1); }
			for (int y = 0; y < h; y++) { SeedCorner(0, y); SeedCorner(w - 1, y); }

			int[] dx4 = { -1, 1, 0, 0 };
			int[] dy4 = { 0, 0, -1, 1 };

			while (queue.Count > 0)
			{
				int cur = queue.Dequeue();
				int cy = cur / w;
				int cx = cur % w;

				for (int d = 0; d < 4; d++)
				{
					int nx = cx + dx4[d];
					int ny = cy + dy4[d];
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

			return LargestComponent(eroded, w, h);
		}

		private bool[] ErodeCircular(bool[] mask, int w, int h, int radius)
		{
			bool[] result = new bool[mask.Length];

			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
				{
					if (!mask[y * w + x]) continue;

					bool fits = true;
				outer:
					for (int dy = -radius; dy <= radius && fits; dy++)
						for (int dx = -radius; dx <= radius && fits; dx++)
						{
							if (dx * dx + dy * dy > radius * radius) continue;
							int nx = x + dx, ny = y + dy;
							if (nx < 0 || nx >= w || ny < 0 || ny >= h || !mask[ny * w + nx])
							{
								fits = false;
								goto outer;
							}
						}

					result[y * w + x] = fits;
				}

			return result;
		}

		private bool[] LargestComponent(bool[] mask, int w, int h)
		{
			int[] labels = new int[mask.Length];
			Array.Fill(labels, -1);
			int nextLabel = 0;
			var sizes = new List<int>();

			int[] dx4 = { -1, 1, 0, 0 };
			int[] dy4 = { 0, 0, -1, 1 };

			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
				{
					int idx = y * w + x;
					if (!mask[idx] || labels[idx] >= 0) continue;

					var q = new Queue<int>();
					q.Enqueue(idx);
					labels[idx] = nextLabel;
					int sz = 0;

					while (q.Count > 0)
					{
						int cur = q.Dequeue(); sz++;
						int cy = cur / w, cx = cur % w;

						for (int d = 0; d < 4; d++)
						{
							int nx = cx + dx4[d], ny = cy + dy4[d];
							if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
							int ni = ny * w + nx;
							if (mask[ni] && labels[ni] < 0)
							{
								labels[ni] = nextLabel;
								q.Enqueue(ni);
							}
						}
					}

					sizes.Add(sz);
					nextLabel++;
				}

			if (nextLabel == 0) return mask;

			int maxLabel = sizes.IndexOf(sizes.Max());
			bool[] result = new bool[mask.Length];
			for (int i = 0; i < mask.Length; i++)
				result[i] = labels[i] == maxLabel;

			return result;
		}

		private byte[] MaskPixels(byte[] pixels, bool[] mask)
		{
			byte[] masked = new byte[pixels.Length];
			for (int i = 0; i < pixels.Length; i++)
				masked[i] = mask[i] ? pixels[i] : (byte)0;
			return masked;
		}

		private byte[] BuildClassMap(byte[] pixels, double[] thresholds)
		{
			double[] T = (double[])thresholds.Clone();
			Array.Sort(T);

			byte[] classMap = new byte[pixels.Length];
			for (int i = 0; i < pixels.Length; i++)
			{
				classMap[i] =
					pixels[i] <= T[0] ? (byte)0 :
					pixels[i] <= T[1] ? (byte)1 :
					pixels[i] <= T[2] ? (byte)2 : (byte)3;
			}

			return classMap;
		}

		private bool[] ExtractTumorMask(byte[] classMap, bool[] brainMask, int w, int h)
		{
			bool[] candidate = new bool[classMap.Length];
			for (int i = 0; i < classMap.Length; i++)
				candidate[i] = classMap[i] == 3 && brainMask[i];

			int candidateCount = candidate.Count(b => b);

			if (candidateCount < w * h * 0.001)
			{
				for (int i = 0; i < classMap.Length; i++)
					candidate[i] = (classMap[i] == 3 || classMap[i] == 2) && brainMask[i];
			}

			bool[] result = LargestComponent(candidate, w, h);

			int resultCount = result.Count(b => b);
			double pct = resultCount * 100.0 / (w * h);

			if (pct > 40 || resultCount < 10)
				return new bool[classMap.Length];

			return result;
		}

		private BitmapSource CreateTumorOverlay(
			byte[] original, bool[] tumorMask, int width, int height)
		{
			byte[] output = new byte[width * height * 4];

			for (int i = 0; i < original.Length; i++)
			{
				int idx = i * 4;
				byte gray = original[i];

				if (tumorMask[i])
				{
					output[idx + 0] = 30;   
					output[idx + 1] = 30;   
					output[idx + 2] = 255;  
					output[idx + 3] = 255;  
				}
				else
				{
					byte dimmed = (byte)(gray * 0.55);
					output[idx + 0] = dimmed;
					output[idx + 1] = dimmed;
					output[idx + 2] = dimmed;
					output[idx + 3] = 255;
				}
			}

			var wb = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
			wb.WritePixels(new Int32Rect(0, 0, width, height), output, width * 4, 0);
			wb.Freeze();
			return wb;
		}

		private BitmapSource MatToBitmapSource(Mat mat)
		{
			int w = mat.Width, h = mat.Height;
			byte[] pixels = new byte[w * h];
			mat.GetArray(out pixels);

			var wb = new WriteableBitmap(w, h, 96, 96, PixelFormats.Gray8, null);
			wb.WritePixels(new Int32Rect(0, 0, w, h), pixels, w, 0);
			wb.Freeze();
			return wb;
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
			=> RunSegmentation(new GPUPSOAdapter(), "GPU PSO");

		private async void BtnBenchmark3_Click(object sender, RoutedEventArgs e)
		{
			if (currentPixels == null)
			{
				MessageBox.Show("Load image first!");
				return;
			}

			txtStatus.Text = "Running 3-run benchmark (Multi-Core PSO)...";
			var times = new double[3];

			for (int i = 0; i < 3; i++)
			{
				var sw = Stopwatch.StartNew();
				new MultiCorePSO(currentPixels).Run(100);
				sw.Stop();
				times[i] = sw.Elapsed.TotalMilliseconds;
			}

			double avg = times.Average();

			txtAvgTime.Text = $"Average Time (3 runs): {avg:F2} ms";
			txtStatus.Text = "Benchmark Completed";

			MessageBox.Show(
				$"3-Run Benchmark Done!\n" +
				$"Run 1: {times[0]:F2} ms\n" +
				$"Run 2: {times[1]:F2} ms\n" +
				$"Run 3: {times[2]:F2} ms\n" +
				$"Average: {avg:F2} ms"
			);
		}

		private void BtnSaveSegmented_Click(object sender, RoutedEventArgs e)
		{
			if (lastSegmentedImage == null)
			{
				MessageBox.Show("No segmented image to save!");
				return;
			}

			var dlg = new SaveFileDialog
			{
				Filter = "PNG Image|*.png|JPEG Image|*.jpg",
				FileName = "segmented_output.png"
			};

			if (dlg.ShowDialog() == true)
			{
				BitmapEncoder encoder = dlg.FilterIndex == 2
					? new JpegBitmapEncoder()
					: new PngBitmapEncoder();

				encoder.Frames.Add(BitmapFrame.Create(lastSegmentedImage));

				using var stream = System.IO.File.Create(dlg.FileName);
				encoder.Save(stream);

				MessageBox.Show("Image saved successfully!");
			}
		}
	}
}