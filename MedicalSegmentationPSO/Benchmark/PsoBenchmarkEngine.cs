using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MedicalSegmentationPSO.Benchmark
{
    public class PsoBenchmarkEngine
    {
        public BenchmarkResult RunOnDataset(
            List<byte[]> dataset,
            IPSOAlgorithm algorithm)
        {
            List<double> imageTimes = new();

            Stopwatch totalSW = Stopwatch.StartNew();

            foreach (var img in dataset)
            {
                Stopwatch sw = Stopwatch.StartNew();

                algorithm.Run(img);

                sw.Stop();

                imageTimes.Add(sw.Elapsed.TotalMilliseconds);
            }

            totalSW.Stop();

            return new BenchmarkResult
            {
                Algorithm = algorithm.Name,
                DatasetSize = dataset.Count,
                TotalTimeMs = totalSW.Elapsed.TotalMilliseconds,
                AverageTimeMs = imageTimes.Average(),
                StdDevMs = CalculateStdDev(imageTimes),
                Timestamp = DateTime.Now
            };
        }

        private double CalculateStdDev(List<double> values)
        {
            double avg = values.Average();

            return Math.Sqrt(
                values.Average(v =>
                    Math.Pow(v - avg, 2)));
        }
    }
}