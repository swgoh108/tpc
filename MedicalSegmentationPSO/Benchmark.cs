using System;
using System.Collections.Generic;
using System.Text;
using BenchmarkDotNet.Attributes;

namespace MedicalSegmentationPSO
{
    public class PsoBenchmarks
    {
        private byte[] pixels;

        [GlobalSetup]
        public void Setup()
        {
            pixels = ImageProcessor.LoadGrayscalePixels(
                Program.TestImage,
                out int w,
                out int h);
        }

        [Benchmark]
        public void SequentialPSO()
        {
            var pso = new PSO(pixels);
            pso.Run(100);
        }

        //[Benchmark]
        //public void ParallelPSO()
        //{
        //    var pso = new ParallelPSO(pixels); // your future implementation
        //    pso.Run(100);
        //}

        //[Benchmark]
        //public void AdaptivePSO()
        //{
        //    var pso = new AdaptivePSO(pixels);
        //    pso.Run(100);
        //}

        //[Benchmark]
        //public void QuantumInspiredPSO()
        //{
        //    var pso = new QuantumPSO(pixels);
        //    pso.Run(100);
        //}

        //[Benchmark]
        //public void HybridPSO()
        //{
        //    var pso = new HybridPSO(pixels);
        //    pso.Run(100);
        //}
    }
}
