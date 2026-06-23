using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MedicalSegmentationPSO
{
    public class MultiCorePSO
    {
        private readonly List<Particle> _swarm;
        private readonly byte[] _pixels;
        private readonly int _dim = 3;
        private readonly Random _rnd = new();

        private double[] _globalBestPosition;
        private double _globalBestFitness = double.MinValue;

        private readonly int _maxThreads;

        public MultiCorePSO(
            byte[] pixels,
            int swarmSize = 30,
            int? maxThreads = null)
        {
            _pixels = pixels;

            _maxThreads = maxThreads ??
                          Math.Max(1,
                          Environment.ProcessorCount - 1);

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
        }

        public double[] Run(
            int iterations = 100,
            bool verbose = false)
        {
            const double w = 0.7;
            const double c1 = 1.5;
            const double c2 = 1.5;

            for (int iter = 0; iter < iterations; iter++)
            {
                var localBests = new ConcurrentBag<(double Fitness, double[] Position)>();

                // ==================================================
                // PARALLEL FITNESS EVALUATION
                // ==================================================
                Parallel.ForEach(
                    _swarm,
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = _maxThreads
                    },
                    particle =>
                    {
                        double fitness =
                            CalculateFitness(particle.Position);

                        if (fitness > particle.BestFitness)
                        {
                            particle.BestFitness = fitness;

                            Array.Copy(
                                particle.Position,
                                particle.BestPosition,
                                _dim);
                        }

                        localBests.Add(
                        (
                            fitness,
                            (double[])particle.Position.Clone()
                        ));
                    });

                // ==================================================
                // GLOBAL BEST REDUCTION
                // ==================================================
                foreach (var candidate in localBests)
                {
                    if (candidate.Fitness > _globalBestFitness)
                    {
                        _globalBestFitness = candidate.Fitness;

                        Array.Copy(
                            candidate.Position,
                            _globalBestPosition,
                            _dim);
                    }
                }

                // ==================================================
                // VELOCITY & POSITION UPDATE
                // ==================================================
                foreach (var particle in _swarm)
                {
                    for (int d = 0; d < _dim; d++)
                    {
                        double r1 = _rnd.NextDouble();
                        double r2 = _rnd.NextDouble();

                        particle.Velocity[d] =
                            w * particle.Velocity[d]
                            + c1 * r1 *
                            (particle.BestPosition[d]
                            - particle.Position[d])
                            + c2 * r2 *
                            (_globalBestPosition[d]
                            - particle.Position[d]);

                        particle.Position[d] =
                            Math.Clamp(
                                particle.Position[d]
                                + particle.Velocity[d],
                                0,
                                255);
                    }
                }

                if (verbose)
                {
                    Console.WriteLine(
                        $"Iter {iter + 1}/{iterations} | " +
                        $"Best Fitness = {_globalBestFitness:F6}");
                }
            }

            double[] result =
                (double[])_globalBestPosition.Clone();

            Array.Sort(result);

            return result;
        }

        // =========================================================
        // OTSU MULTI-THRESHOLD FITNESS
        // =========================================================
        private double CalculateFitness(double[] thresholds)
        {
            double[] sorted =
                (double[])thresholds.Clone();

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
                    means[i] =
                        sums[i] / weights[i];

                    globalMean +=
                        means[i] *
                        (weights[i] / totalPixels);
                }
            }

            double betweenClassVariance = 0;

            for (int i = 0; i < 4; i++)
            {
                if (weights[i] > 0)
                {
                    betweenClassVariance +=
                        (weights[i] / totalPixels)
                        *
                        Math.Pow(
                            means[i] - globalMean,
                            2);
                }
            }

            return betweenClassVariance;
        }
    }
}