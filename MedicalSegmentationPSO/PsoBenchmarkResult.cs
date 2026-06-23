using System;
using System.Collections.Generic;
using System.Text;

namespace MedicalSegmentationPSO
{
    public class PsoBenchmarkResult
    {
        public string Method { get; set; }
        public double AverageFitness { get; set; }
        public long AverageTimeMs { get; set; }
        public double[] BestThresholds { get; set; }
    }
}
