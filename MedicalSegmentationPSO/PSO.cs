using System;
using System.Collections.Generic;

namespace MedicalSegmentationPSO
{
    // ─────────────────────────────────────────────
    // Particle – one candidate solution [T1, T2, T3]
    // ─────────────────────────────────────────────
    public class Particle
    {
        public double[] Position;     // threshold values
        public double[] Velocity;
        public double[] BestPosition;
        public double BestFitness = double.MinValue;

        public Particle(int dim)
        {
            Position = new double[dim];
            Velocity = new double[dim];
            BestPosition = new double[dim];
        }
    }

    // ─────────────────────────────────────────────
    // PSO – finds 3 optimal Otsu-style thresholds
    // Fitness = Between-class variance (maximise)
    // ─────────────────────────────────────────────
    public class PSO
    {
        private readonly List<Particle> _swarm;
        private readonly byte[] _pixels;
        private readonly int _dim = 3;
        private readonly Random _rnd = new();

        private double[] _globalBestPosition;
        private double _globalBestFitness = double.MinValue;

        public PSO(byte[] pixels, int swarmSize = 30)
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
                    p.Velocity[d] = (_rnd.NextDouble() - 0.5) * 10;
                    p.BestPosition[d] = p.Position[d];
                }
                _swarm.Add(p);
            }
        }

        /// <summary>Runs PSO and returns the 3 best threshold values (sorted).</summary>
        public double[] Run(int iterations = 100, bool verbose = false)
        {
            const double w = 0.7;
            const double c1 = 1.5;
            const double c2 = 1.5;

            for (int iter = 0; iter < iterations; iter++)
            {
                // ── evaluate ──
                foreach (var p in _swarm)
                {
                    double fitness = CalculateFitness(p.Position);

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

                // ── update velocity & position ──
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
                        $"  Iter {iter + 1,3}/{iterations} | " +
                        $"Best Fitness: {_globalBestFitness:F6}");
            }

            double[] result = (double[])_globalBestPosition.Clone();
            Array.Sort(result);
            return result;
        }

        // ── Between-class variance (Otsu criterion, 4 classes) ──
        private double CalculateFitness(double[] T)
        {
            double[] sorted = (double[])T.Clone();
            Array.Sort(sorted);

            double[] w = new double[4];
            double[] sum = new double[4];

            foreach (byte px in _pixels)
            {
                int k =
                    px <= sorted[0] ? 0 :
                    px <= sorted[1] ? 1 :
                    px <= sorted[2] ? 2 : 3;

                w[k]++;
                sum[k] += px;
            }

            double total = _pixels.Length;
            double muT = 0;
            double[] mu = new double[4];

            for (int i = 0; i < 4; i++)
                if (w[i] > 0)
                {
                    mu[i] = sum[i] / w[i];
                    muT += mu[i] * (w[i] / total);
                }

            double fitness = 0;
            for (int i = 0; i < 4; i++)
                if (w[i] > 0)
                    fitness += (w[i] / total) * Math.Pow(mu[i] - muT, 2);

            return fitness;
        }
    }
}