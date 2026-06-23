using System;
using System.Collections.Generic;
using System.Text;

namespace MedicalSegmentationPSO.Benchmark
{
    public interface IPSOAlgorithm
    {
        string Name { get; }

        double[] Run(byte[] pixels);
    }
}
