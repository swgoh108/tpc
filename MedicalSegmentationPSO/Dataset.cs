//Dataset.cs
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

using TorchSharp;

using static TorchSharp.torch;

using OpenCvSharp;

namespace MedicalSegmentationPSO
{
    public class Batch
    {
        public Tensor Images { get; set; }
        public Tensor Labels { get; set; }
    }

    public static class DatasetLoaderTorch
    {
        private const int ImageSize = 224;

        public static List<(Tensor Image, Tensor Label)> Load(string rootDir)
        {
            var dataset =
                new List<(Tensor Image, Tensor Label)>();

            Console.WriteLine(
                $"Loading Dataset: {rootDir}");

            for (int classIdx = 0;
                 classIdx < TumorLabels.Classes.Length;
                 classIdx++)
            {
                string className =
                    TumorLabels.Classes[classIdx];

                string classDir =
                    Path.Combine(
                        rootDir,
                        className);

                if (!Directory.Exists(classDir))
                {
                    Console.WriteLine(
                        $"Missing Folder: {classDir}");

                    continue;
                }

                int count = 0;

                foreach (string file in Directory.GetFiles(
                             classDir,
                             "*.*",
                             SearchOption.TopDirectoryOnly))
                {
                    try
                    {
                        using var mat =
                            Cv2.ImRead(
                                file,
                                ImreadModes.Grayscale);

                        if (mat.Empty())
                            continue;

                        using var resized =
                            new Mat();

                        Cv2.Resize(
                            mat,
                            resized,
                            new OpenCvSharp.Size(
                                ImageSize,
                                ImageSize));

                        byte[] pixels =
                            new byte[
                                ImageSize *
                                ImageSize];

                        Marshal.Copy(
                            resized.Data,
                            pixels,
                            0,
                            pixels.Length);

                        float[] data =
                            new float[
                                pixels.Length];

                        for (int i = 0;
                             i < pixels.Length;
                             i++)
                        {
                            data[i] =
                                pixels[i] / 255f;
                        }

                        Tensor image =
                            torch.tensor(
                                data,
                                dtype:
                                ScalarType.Float32)
                            .reshape(
                                1,
                                ImageSize,
                                ImageSize);

                        Tensor label =
                            torch.tensor(
                                new long[]
                                {
                                    classIdx
                                },
                                dtype:
                                ScalarType.Int64);

                        dataset.Add(
                            (image, label));

                        count++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(
                            $"Skip: {file}");

                        Console.WriteLine(
                            ex.Message);
                    }
                }

                Console.WriteLine(
                    $"{className}: {count} images");
            }

            Console.WriteLine(
                $"Total Samples Loaded = {dataset.Count}");

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
            if (shuffle)
            {
                var rng =
                    new Random();

                dataset =
                    dataset
                    .OrderBy(
                        x => rng.Next())
                    .ToList();
            }

            for (int i = 0;
                 i < dataset.Count;
                 i += batchSize)
            {
                var batchSamples =
                    dataset
                    .Skip(i)
                    .Take(batchSize)
                    .ToList();

                Tensor[] images =
                    batchSamples
                    .Select(x => x.Image)
                    .ToArray();

                long[] labels =
                    batchSamples
                    .Select(
                        x => x.Label.item<long>())
                    .ToArray();

                yield return new Batch
                {
                    Images =
                        torch.stack(images),

                    Labels =
                        torch.tensor(
                            labels,
                            dtype:
                            ScalarType.Int64)
                };
            }
        }
    }
}