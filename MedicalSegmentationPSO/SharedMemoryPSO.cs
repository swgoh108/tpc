using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace MedicalSegmentationPSO
{
    public class SharedMemoryPSO
    {
        private readonly List<Particle> _swarm;
        private readonly byte[] _pixels;
        private readonly int _dim = 3;
        private readonly int _maxThreads;

        private readonly double[] _globalBestPosition;
        private double _globalBestFitness = double.MinValue;

        private readonly object _globalBestLock = new();

        private int _evaluatedParticles = 0;
        private int _globalBestUpdates = 0;

        public SharedMemoryPSO(byte[] pixels, int swarmSize = 30, int? maxThreads = null)
        {
            _pixels = pixels;
            _maxThreads = maxThreads ?? Math.Max(1, Environment.ProcessorCount - 1);

            _swarm = new List<Particle>(swarmSize);
            _globalBestPosition = new double[_dim];

            Random rnd = new Random();

            for (int i = 0; i < swarmSize; i++)
            {
                Particle p = new Particle(_dim);

                for (int d = 0; d < _dim; d++)
                {
                    p.Position[d] = rnd.Next(1, 255);
                    p.Velocity[d] = (rnd.NextDouble() - 0.5) * 10.0;
                    p.BestPosition[d] = p.Position[d];
                }

                _swarm.Add(p);
            }
        }

        public double[] Run(int iterations = 100, bool verbose = false)
        {
            const double w = 0.7;
            const double c1 = 1.5;
            const double c2 = 1.5;

            for (int iter = 0; iter < iterations; iter++)
            {
                ThreadLocal<(double Fitness, double[] Position)> localBest =
                    new ThreadLocal<(double Fitness, double[] Position)>(
                        () => (double.MinValue, new double[_dim]),
                        trackAllValues: true);

                Parallel.For(
                    0,
                    _swarm.Count,
                    new ParallelOptions { MaxDegreeOfParallelism = _maxThreads },
                    i =>
                    {
                        Particle p = _swarm[i];

                        double fitness = CalculateFitness(p.Position);

                        Interlocked.Increment(ref _evaluatedParticles);

                        if (fitness > p.BestFitness)
                        {
                            p.BestFitness = fitness;
                            Array.Copy(p.Position, p.BestPosition, _dim);
                        }

                        if (fitness > localBest.Value.Fitness)
                        {
                            localBest.Value =
                                (fitness, (double[])p.Position.Clone());
                        }
                    });

                foreach (var candidate in localBest.Values)
                {
                    lock (_globalBestLock)
                    {
                        if (candidate.Fitness > _globalBestFitness)
                        {
                            _globalBestFitness = candidate.Fitness;
                            Array.Copy(candidate.Position, _globalBestPosition, _dim);

                            Interlocked.Increment(ref _globalBestUpdates);
                        }
                    }
                }

                double[] gBestSnapshot;

                lock (_globalBestLock)
                {
                    gBestSnapshot = (double[])_globalBestPosition.Clone();
                }

                Parallel.For(
                    0,
                    _swarm.Count,
                    new ParallelOptions { MaxDegreeOfParallelism = _maxThreads },
                    i =>
                    {
                        Particle p = _swarm[i];

                        Random rnd = new Random(
                            Environment.TickCount + i * 997);

                        for (int d = 0; d < _dim; d++)
                        {
                            double r1 = rnd.NextDouble();
                            double r2 = rnd.NextDouble();

                            p.Velocity[d] =
                                w * p.Velocity[d]
                                + c1 * r1 * (p.BestPosition[d] - p.Position[d])
                                + c2 * r2 * (gBestSnapshot[d] - p.Position[d]);

                            p.Position[d] =
                                Math.Clamp(p.Position[d] + p.Velocity[d], 0, 255);
                        }
                    });

                if (verbose)
                {
                    Console.WriteLine(
                        $"[Shared Memory] Iter {iter + 1}/{iterations} | " +
                        $"Best Fitness = {_globalBestFitness:F6}");
                }
            }

            double[] result;

            lock (_globalBestLock)
            {
                result = (double[])_globalBestPosition.Clone();
            }

            Array.Sort(result);
            return result;
        }

        private double CalculateFitness(double[] thresholds)
        {
            double[] sorted = (double[])thresholds.Clone();
            Array.Sort(sorted);

            double[] weights = new double[4];
            double[] sums = new double[4];

            foreach (byte pixel in _pixels)
            {
                int cls =
                    pixel <= sorted[0] ? 0 :
                    pixel <= sorted[1] ? 1 :
                    pixel <= sorted[2] ? 2 : 3;

                weights[cls]++;
                sums[cls] += pixel;
            }

            double totalPixels = _pixels.Length;
            double[] means = new double[4];
            double globalMean = 0;

            for (int i = 0; i < 4; i++)
            {
                if (weights[i] > 0)
                {
                    means[i] = sums[i] / weights[i];
                    globalMean += means[i] * (weights[i] / totalPixels);
                }
            }

            double variance = 0;

            for (int i = 0; i < 4; i++)
            {
                if (weights[i] > 0)
                {
                    variance +=
                        (weights[i] / totalPixels)
                        * Math.Pow(means[i] - globalMean, 2);
                }
            }

            return variance;
        }
    }
}