using System;
using System.Collections.Generic;
using System.Linq;

namespace MedicalSegmentationPSO
{
    public class PLINQPSO
    {
        private readonly List<Particle> _swarm;
        private readonly byte[] _pixels;
        private readonly int _dim = 3;
        private readonly Random _rnd = new();

        private double[] _globalBestPosition;
        private double _globalBestFitness = double.MinValue;

        private readonly int _maxThreads;

        public PLINQPSO(byte[] pixels, int swarmSize = 30, int? maxThreads = null)
        {
            _pixels = pixels;
            _maxThreads = maxThreads ?? Math.Max(1, Environment.ProcessorCount - 1);
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

        public double[] Run(int iterations = 100, bool verbose = false)
        {
            const double w = 0.7;
            const double c1 = 1.5;
            const double c2 = 1.5;

            for (int iter = 0; iter < iterations; iter++)
            {
                // ── PLINQ PARALLEL FITNESS EVALUATION ──
                var evaluated = _swarm
                    .AsParallel()
                    .WithDegreeOfParallelism(_maxThreads)
                    .Select(p =>
                    {
                        double fitness = CalculateFitness(p.Position);
                        return (Particle: p, Fitness: fitness);
                    })
                    .ToArray();

                // ── update personal & global bests (sequential reduction) ──
                foreach (var (p, fitness) in evaluated)
                {
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

                // ── velocity & position update ──
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

        // ── Otsu 4-class between-class variance (identical to PSO.cs) ──
        private double CalculateFitness(double[] thresholds)
        {
            double[] sorted = (double[])thresholds.Clone();
            Array.Sort(sorted);

            double[] weights = new double[4];
            double[] sums = new double[4];

            foreach (byte px in _pixels)
            {
                int k = px <= sorted[0] ? 0 :
                        px <= sorted[1] ? 1 :
                        px <= sorted[2] ? 2 : 3;
                weights[k]++;
                sums[k] += px;
            }

            double total = _pixels.Length;
            double muT = 0;
            double[] mu = new double[4];

            for (int i = 0; i < 4; i++)
                if (weights[i] > 0)
                {
                    mu[i] = sums[i] / weights[i];
                    muT += mu[i] * (weights[i] / total);
                }

            double fitness = 0;
            for (int i = 0; i < 4; i++)
                if (weights[i] > 0)
                    fitness += (weights[i] / total) * Math.Pow(mu[i] - muT, 2);

            return fitness;
        }
    }
}
