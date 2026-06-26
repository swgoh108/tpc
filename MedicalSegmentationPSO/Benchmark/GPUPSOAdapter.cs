using System;
using System.Collections.Generic;
using System.Text;

namespace MedicalSegmentationPSO.Benchmark
{
    public class GPUPSOAdapter : IPSOAlgorithm
    {
        public string Name => "GPU PSO (ILGPU)";

        public double[] Run(byte[] pixels)
        {
            using var pso = new GPUPSO(pixels);
            return pso.Run(100);
        }
    }
}
