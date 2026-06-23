using System;

namespace MedicalSegmentationPSO.Benchmark
{
    public class BenchmarkResult
    {
        public string Algorithm { get; set; }

        public int DatasetSize { get; set; }

        public double TotalTimeMs { get; set; }

        public double AverageTimeMs { get; set; }

        public double StdDevMs { get; set; }

        public double Speedup { get; set; }

        public int NumberOfCores { get; set; }

        public double Efficiency { get; set; }

        public DateTime Timestamp { get; set; }
    }
}