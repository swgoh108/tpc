using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using static Program;

namespace MedicalSegmentationPSO
{
    public static class SequentialPsoRunner
    {
        public static PsoBenchmarkResult Run(byte[] pixels, bool forceRun = false)
        {
            var cached = ResultCache.Load("sequential_pso");
            if (cached != null && !forceRun)
            {
                Console.WriteLine("[CACHE] Loaded Sequential PSO result");
                return cached;
            }

            Console.WriteLine("[RUN] Sequential PSO executing...");

            var sw = Stopwatch.StartNew();

            var pso = new PSO(pixels);
            var best = pso.Run(100);

            sw.Stop();

            var result = new PsoBenchmarkResult
            {
                Method = "Sequential PSO",
                AverageFitness = 0, // optional if you want to extend
                AverageTimeMs = sw.ElapsedMilliseconds,
                BestThresholds = best
            };

            ResultCache.Save("sequential_pso", result);

            return result;
        }
    }
}
