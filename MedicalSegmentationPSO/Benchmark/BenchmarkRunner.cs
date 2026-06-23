using System;
using System.Collections.Generic;
using System.IO;
using MedicalSegmentationPSO.Benchmark;

namespace MedicalSegmentationPSO
{
    public static class BenchmarkPSORunner
    {
        public static void Run()
        {
            Console.WriteLine(
                "=== DATASET BENCHMARK ===");

            string datasetPath =
                @"C:\Users\user\BrainTumorMRIDataset\Testing";

            List<byte[]> dataset =LoadDataset(datasetPath);

            Console.WriteLine($"Images Loaded: {dataset.Count}");

            var engine =
                new PsoBenchmarkEngine();

            var algorithms =
                new List<IPSOAlgorithm>
                {
                    new SequentialPSOAdapter(),
                    new MultiCorePSOAdapter(),
                    //new ParallelPSOAdapter(),
                    //new GPUPSOAdapter(),
                };

            BenchmarkResult baseline = null;

            foreach (var algorithm in algorithms)
            {
                Console.WriteLine(
                    $"\nRunning {algorithm.Name}...");

                var result =
                    engine.RunOnDataset(
                        dataset,
                        algorithm);

                if (baseline == null)
                {
                    baseline = result;

                    result.Speedup = 1;
                    result.NumberOfCores = 1;
                    result.Efficiency = 1;
                }
                else
                {
                    result.Speedup =
                        baseline.TotalTimeMs /
                        result.TotalTimeMs;

                    result.NumberOfCores =
                        Environment.ProcessorCount;

                    result.Efficiency =
                        result.Speedup /
                        result.NumberOfCores;
                }

                PrintResult(result);

                JsonResultWriter.Save(result);
            }
        }

        private static void PrintResult(
            BenchmarkResult result)
        {
            Console.WriteLine(
                $"Algorithm : {result.Algorithm}");

            Console.WriteLine(
                $"Dataset Size : {result.DatasetSize}");

            Console.WriteLine(
                $"Total Time : {result.TotalTimeMs:F2} ms");

            Console.WriteLine(
                $"Average Time : {result.AverageTimeMs:F2} ms");

            Console.WriteLine(
                $"StdDev : {result.StdDevMs:F2}");

            Console.WriteLine(
                $"Speedup : {result.Speedup:F2}");

            Console.WriteLine(
                $"Efficiency : {result.Efficiency:F2}");
        }

        private static List<byte[]> LoadDataset(
            string path)
        {
            var dataset =
                new List<byte[]>();

            foreach (string file in
                     Directory.GetFiles(
                         path,
                         "*.*",
                         SearchOption.AllDirectories))
            {
                try
                {
                    var pixels =
                        ImageProcessor
                        .LoadGrayscalePixels(
                            file,
                            out _,
                            out _);

                    dataset.Add(pixels);
                }
                catch
                {
                }
            }

            return dataset;
        }
    }
}