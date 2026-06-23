// DatasetSegmentation.cs
using System;
using System.IO;

namespace MedicalSegmentationPSO
{
    /// <summary>
    /// Applies PSO-based threshold segmentation to every image in a source
    /// dataset folder and writes the result to a destination folder,
    /// preserving the per-class sub-folder structure.
    /// </summary>
    public static class DatasetSegmentation
    {
        private static readonly string[] SupportedExtensions =
            { ".jpg", ".jpeg", ".png", ".bmp" };

        // =========================
        // PROCESS
        // =========================
        public static void Process(string sourceRoot, string destRoot, int psoIterations = 50)
        {
            Console.WriteLine($"  Source : {sourceRoot}");
            Console.WriteLine($"  Dest   : {destRoot}");
            Console.WriteLine($"  PSO iterations per image: {psoIterations}");

            string[] classDirs = Directory.GetDirectories(sourceRoot);

            foreach (string classDir in classDirs)
            {
                string className = Path.GetFileName(classDir);
                string destClass = Path.Combine(destRoot, className);
                Directory.CreateDirectory(destClass);

                string[] files = Directory.GetFiles(classDir);
                int saved = 0;

                foreach (string file in files)
                {
                    string ext = Path.GetExtension(file).ToLowerInvariant();
                    if (Array.IndexOf(SupportedExtensions, ext) < 0) continue;

                    try
                    {
                        // Load as grayscale for PSO threshold optimisation
                        byte[] pixels = ImageProcessor.LoadGrayscalePixels(
                            file, out int w, out int h);

                        // Run PSO to find optimal thresholds
                        double[] thresholds = PsoSegmentation.FindThresholds(
                            pixels, iterations: psoIterations);

                        // Apply thresholds → segmented label image
                        byte[] segmented = ApplyThresholds(pixels, thresholds);

                        // Save segmented image
                        string destFile = Path.Combine(
                            destClass, Path.GetFileName(file));

                        SaveGrayscale(segmented, w, h, destFile);
                        saved++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  [WARN] Skipping {file}: {ex.Message}");
                    }
                }

                Console.WriteLine($"  [{className}] {saved} images segmented.");
            }

            Console.WriteLine("  Segmentation complete.");
        }

        // =========================
        // APPLY THRESHOLDS
        // =========================
        // Maps each pixel to one of 4 quantised intensity levels based on PSO thresholds
        private static byte[] ApplyThresholds(byte[] pixels, double[] T)
        {
            double[] sorted = (double[])T.Clone();
            Array.Sort(sorted);

            // 4 output levels (evenly spaced for visualisation)
            byte[] levels = { 0, 85, 170, 255 };
            byte[] result = new byte[pixels.Length];

            for (int i = 0; i < pixels.Length; i++)
            {
                int k =
                    pixels[i] <= sorted[0] ? 0 :
                    pixels[i] <= sorted[1] ? 1 :
                    pixels[i] <= sorted[2] ? 2 : 3;

                result[i] = levels[k];
            }

            return result;
        }

        // =========================
        // SAVE GRAYSCALE
        // =========================
        // Delegates to ImageProcessor.SaveGrayscale (uses SkiaSharp — MIT licensed)
        private static void SaveGrayscale(byte[] pixels, int w, int h, string path)
        {
            // Change extension to .png — lossless format suits segmentation masks
            string pngPath = System.IO.Path.ChangeExtension(path, ".png");
            ImageProcessor.SaveGrayscale(pixels, w, h, pngPath);
        }
    }

    // =========================
    // PSO SEGMENTATION CORE
    // =========================
    /// <summary>
    /// Particle Swarm Optimisation that finds 3 intensity thresholds
    /// maximising Otsu-style between-class variance across 4 classes.
    /// </summary>
    public static class PsoSegmentation
    {
        private const int Particles = 20;
        private const int Dimensions = 3;   // 3 thresholds → 4 segments

        public static double[] FindThresholds(byte[] pixels, int iterations = 50)
        {
            var rng = new Random();

            // Build histogram once
            double[] hist = BuildHistogram(pixels);

            // Initialise particles
            double[][] pos = new double[Particles][];
            double[][] vel = new double[Particles][];
            double[][] bestPos = new double[Particles][];
            double[] bestFit = new double[Particles];

            double[] gBestPos = new double[Dimensions];
            double gBestFit = double.MinValue;

            for (int i = 0; i < Particles; i++)
            {
                pos[i] = RandomThresholds(rng);
                vel[i] = new double[Dimensions];
                bestPos[i] = (double[])pos[i].Clone();
                bestFit[i] = Fitness(pos[i], hist);

                if (bestFit[i] > gBestFit)
                {
                    gBestFit = bestFit[i];
                    gBestPos = (double[])pos[i].Clone();
                }
            }

            // PSO main loop
            const double w = 0.5;   // inertia
            const double c1 = 1.5;   // cognitive
            const double c2 = 1.5;   // social

            for (int iter = 0; iter < iterations; iter++)
            {
                for (int i = 0; i < Particles; i++)
                {
                    for (int d = 0; d < Dimensions; d++)
                    {
                        double r1 = rng.NextDouble();
                        double r2 = rng.NextDouble();

                        vel[i][d] = w * vel[i][d]
                                  + c1 * r1 * (bestPos[i][d] - pos[i][d])
                                  + c2 * r2 * (gBestPos[d] - pos[i][d]);

                        pos[i][d] = Math.Clamp(pos[i][d] + vel[i][d], 0, 255);
                    }

                    double fit = Fitness(pos[i], hist);

                    if (fit > bestFit[i])
                    {
                        bestFit[i] = fit;
                        bestPos[i] = (double[])pos[i].Clone();

                        if (fit > gBestFit)
                        {
                            gBestFit = fit;
                            gBestPos = (double[])pos[i].Clone();
                        }
                    }
                }
            }

            Array.Sort(gBestPos);
            return gBestPos;
        }

        // =========================
        // HISTOGRAM
        // =========================
        private static double[] BuildHistogram(byte[] pixels)
        {
            double[] hist = new double[256];
            foreach (byte p in pixels) hist[p]++;
            for (int i = 0; i < 256; i++) hist[i] /= pixels.Length;
            return hist;
        }

        // =========================
        // FITNESS  (between-class variance)
        // =========================
        private static double Fitness(double[] T, double[] hist)
        {
            double[] sorted = (double[])T.Clone();
            Array.Sort(sorted);

            int[] bounds = {
                0,
                (int)sorted[0],
                (int)sorted[1],
                (int)sorted[2],
                255
            };

            double totalMean = 0;
            for (int i = 0; i < 256; i++) totalMean += i * hist[i];

            double variance = 0;

            for (int k = 0; k < 4; k++)
            {
                int lo = k == 0 ? 0 : bounds[k] + 1;
                int hi = k == 3 ? 255 : bounds[k + 1];

                double prob = 0, mean = 0;
                for (int i = lo; i <= hi; i++)
                {
                    prob += hist[i];
                    mean += i * hist[i];
                }

                if (prob > 0)
                {
                    mean /= prob;
                    variance += prob * Math.Pow(mean - totalMean, 2);
                }
            }

            return variance;
        }

        // =========================
        // RANDOM INIT
        // =========================
        private static double[] RandomThresholds(Random rng)
        {
            double[] t = {
                rng.NextDouble() * 255,
                rng.NextDouble() * 255,
                rng.NextDouble() * 255
            };
            Array.Sort(t);
            return t;
        }
    }
}