// DatasetLoaderTorch.cs
using System;
using System.Collections.Generic;
using System.IO;
using TorchSharp;
using static TorchSharp.torch;

namespace MedicalSegmentationPSO
{
    /// <summary>
    /// Loads a folder-based image dataset into (Image, Label) tensor pairs.
    ///
    /// Expected folder structure (one sub-folder per class, matching TumorLabels.Classes):
    ///   SegmentedDataset/
    ///     glioma/        → label 0
    ///     meningioma/    → label 1
    ///     pituitary/     → label 2
    ///     notumor/       → label 3
    /// </summary>
    public class Batch
    {
        public Tensor Images { get; set; }
        public Tensor Labels { get; set; }
    }
    public static class DatasetLoaderTorch
    {
        public static List<(Tensor Image, Tensor Label)> Load(string rootDir)
        {
            var dataset = new List<(Tensor, Tensor)>();

            for (int classIdx = 0; classIdx < TumorLabels.Classes.Length; classIdx++)
            {
                string className = TumorLabels.Classes[classIdx];
                string classDir = Path.Combine(rootDir, className);

                if (!Directory.Exists(classDir))
                    continue;

                foreach (var file in Directory.GetFiles(classDir))
                {
                    try
                    {
                        using var mat = OpenCvSharp.Cv2.ImRead(file, OpenCvSharp.ImreadModes.Grayscale);

                        if (mat.Empty()) continue;

                        mat.ConvertTo(mat, OpenCvSharp.MatType.CV_32FC1, 1.0 / 255);

                        // Convert directly → Tensor (SAFE)
                        var tensor = torch.tensor(mat.Data)
                            .reshape(mat.Rows, mat.Cols)
                            .unsqueeze(0)   // C
                            .unsqueeze(0);  // N

                        var label = torch.tensor(new long[] { classIdx });

                        dataset.Add((tensor, label));
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Skip {file}: {ex.Message}");
                    }
                }

                Console.WriteLine($"Loaded {className}");
            }

            return dataset;
        }
    }
    public static class TorchDataLoader
    {
        public static IEnumerable<Batch> CreateBatches(
            List<(Tensor Image, Tensor Label)> dataset,
            int batchSize,
            bool shuffle = true)
        {
            var rng = new Random();

            if (shuffle)
            {
                dataset = dataset
                    .OrderBy(_ => rng.Next())
                    .ToList();
            }

            for (int i = 0; i < dataset.Count; i += batchSize)
            {
                var batchSamples = dataset
                    .Skip(i)
                    .Take(batchSize)
                    .ToList();

                var imageList = new List<Tensor>();
                var labelList = new List<long>();

                foreach (var (img, lbl) in batchSamples)
                {
                    imageList.Add(img.squeeze(0));

                    labelList.Add(lbl.item<long>());
                }

                Tensor images = torch.stack(imageList.ToArray());

                Tensor labels = torch.tensor(
                    labelList.ToArray(),
                    dtype: ScalarType.Int64);

                yield return new Batch
                {
                    Images = images,
                    Labels = labels
                };
            }
        }
    }
}