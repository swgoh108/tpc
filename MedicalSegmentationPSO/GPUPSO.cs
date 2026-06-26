using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;

namespace MedicalSegmentationPSO
{
    public class GPUPSO : IDisposable
    {
        private readonly List<Particle> _swarm;
        private readonly byte[] _pixels;
        private readonly int _dim = 3;
        private readonly Random _rnd = new();

        private double[] _globalBestPosition;
        private double _globalBestFitness = double.MinValue;

        private readonly Context _context;
        private readonly Accelerator _accelerator;
        private readonly MemoryBuffer1D<byte, Stride1D.Dense> _gpuPixels;

        // allocate once, reuse every fitness call
        private readonly MemoryBuffer1D<int, Stride1D.Dense> _counts;
        private readonly MemoryBuffer1D<long, Stride1D.Dense> _sums;

        private readonly Action<Index1D, ArrayView<byte>, int, int, int,
                      ArrayView<int>, ArrayView<long>> _kernel;

        public GPUPSO(byte[] pixels, int swarmSize = 30)
        {
            _pixels = pixels;
            _swarm = new List<Particle>(swarmSize);
            _globalBestPosition = new double[_dim];

            for (int i = 0; i < swarmSize; i++)
            {
                var p = new Particle(_dim);
                for (int d = 0; d < _dim; d++)
                {
                    p.Position[d] = _rnd.Next(1, 255);
                    p.Velocity[d] = (_rnd.NextDouble() - 0.5) * 10.0;
                    p.BestPosition[d] = p.Position[d];
                }
                _swarm.Add(p);
            }

            // Pick CUDA if available, else fall back to CPU.
            _context = Context.Create(builder => builder.Cuda());

            var cudaDevice = _context.GetCudaDevice(0);
            _accelerator = cudaDevice.CreateAccelerator(_context);

            _gpuPixels = _accelerator.Allocate1D(_pixels);
            _counts = _accelerator.Allocate1D<int>(4);
            _sums = _accelerator.Allocate1D<long>(4);

            _kernel = _accelerator.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<byte>, int, int, int,
                ArrayView<int>, ArrayView<long>>(HistogramKernel);
        }

        public double[] Run(int iterations = 100, bool verbose = false)
        {
            const double w = 0.7;
            const double c1 = 1.5;
            const double c2 = 1.5;

            for (int iter = 0; iter < iterations; iter++)
            {
                foreach (var p in _swarm)
                {
                    double fitness = CalculateFitnessGPU(p.Position);

                    if (fitness > p.BestFitness)
                    {
                        p.BestFitness = fitness;
                        Array.Copy(p.Position, p.BestPosition, _dim);
                    }
                    if (fitness > _globalBestFitness)
                    {
                        _globalBestFitness = fitness;
                        Array.Copy(p.Position, _globalBestPosition, _dim);
                    }
                }

                foreach (var p in _swarm)
                {
                    for (int d = 0; d < _dim; d++)
                    {
                        double r1 = _rnd.NextDouble();
                        double r2 = _rnd.NextDouble();

                        p.Velocity[d] =
                            w * p.Velocity[d] +
                            c1 * r1 * (p.BestPosition[d] - p.Position[d]) +
                            c2 * r2 * (_globalBestPosition[d] - p.Position[d]);

                        p.Position[d] = Math.Clamp(
                            p.Position[d] + p.Velocity[d], 0, 255);
                    }
                }

                if (verbose)
                    Console.WriteLine(
                        $"Iter {iter + 1}/{iterations} | " +
                        $"Best Fitness = {_globalBestFitness:F6}");
            }

            double[] result = (double[])_globalBestPosition.Clone();
            Array.Sort(result);
            return result;
        }

        private double CalculateFitnessGPU(double[] thresholds)
        {
            double[] sorted = (double[])thresholds.Clone();
            Array.Sort(sorted);

            int t0 = (int)sorted[0];
            int t1 = (int)sorted[1];
            int t2 = (int)sorted[2];

            _counts.MemSetToZero();
            _sums.MemSetToZero();

            _kernel((int)_gpuPixels.Length, _gpuPixels.View,
                    t0, t1, t2, _counts.View, _sums.View);
            _accelerator.Synchronize();

            int[] w = _counts.GetAsArray1D();
            long[] s = _sums.GetAsArray1D();

            double total = _pixels.Length;
            double muT = 0;
            double[] mu = new double[4];

            for (int i = 0; i < 4; i++)
                if (w[i] > 0)
                {
                    mu[i] = (double)s[i] / w[i];
                    muT += mu[i] * (w[i] / total);
                }

            double fitness = 0;
            for (int i = 0; i < 4; i++)
                if (w[i] > 0)
                    fitness += (w[i] / total) * Math.Pow(mu[i] - muT, 2);

            return fitness;
        }

        // one GPU thread per pixel
        private static void HistogramKernel(
            Index1D i, ArrayView<byte> pixels,
            int t0, int t1, int t2,
            ArrayView<int> counts, ArrayView<long> sums)
        {
            byte px = pixels[i];
            int k = px <= t0 ? 0 : px <= t1 ? 1 : px <= t2 ? 2 : 3;
            Atomic.Add(ref counts[k], 1);
            Atomic.Add(ref sums[k], (long)px);
        }

        public void Dispose()
        {
            _sums.Dispose();
            _counts.Dispose();
            _gpuPixels.Dispose();
            _accelerator.Dispose();
            _context.Dispose();
        }
    }

}
