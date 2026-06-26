using System;
using System.Collections.Generic;
using System.Text;

namespace MedicalSegmentationPSO.Benchmark
{
    public class PLINQPSOAdapter : IPSOAlgorithm
    {
        public string Name => "PLINQ PSO";

        public double[] Run(byte[] pixels)
        {
            var pso = new PLINQPSO(pixels);
            return pso.Run(100);
        }
    }
}
