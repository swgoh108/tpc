// SegmentationFeatures.cs
// FIX 6: This code was incorrectly merged into Program.cs in the original.
//        It must live in its own file (or at least outside the top-level
//        Program class scope) so the namespace and classes compile correctly.
using System;

namespace MedicalSegmentationPSO
{
    /// <summary>Simple statistics derived from the PSO segmentation output.</summary>
    public class SegmentationFeatures
    {
        public int BackgroundPixels { get; init; }
        public int LowIntPixels { get; init; }
        public int MidIntPixels { get; init; }
        public int TumorPixels { get; init; }   // brightest class
        public int TotalPixels { get; init; }

        public double BackgroundPct { get; init; }
        public double LowIntPct { get; init; }
        public double MidIntPct { get; init; }
        public double TumorPercentage { get; init; }

        public void Print()
        {
            Console.WriteLine($"  Background    : {BackgroundPixels,7} px  ({BackgroundPct:F2}%)");
            Console.WriteLine($"  Low-intensity : {LowIntPixels,7} px  ({LowIntPct:F2}%)");
            Console.WriteLine($"  Mid-intensity : {MidIntPixels,7} px  ({MidIntPct:F2}%)");
            Console.WriteLine($"  Tumor region  : {TumorPixels,7} px  ({TumorPercentage:F2}%)");
        }
    }

    public static class FeatureExtractor
    {
        /// <summary>
        /// Partitions pixels into the 4 PSO-defined bands and returns counts.
        /// The *highest* intensity band is treated as the potential tumour region.
        /// </summary>
        public static SegmentationFeatures Extract(byte[] pixels, double[] T)
        {
            // FIX 7: Clone and sort T so the original thresholds array is not mutated.
            //        The original code sorts in-place which would corrupt caller's data
            //        if T is reused across multiple calls.
            double[] sorted = (double[])T.Clone();
            Array.Sort(sorted);

            // Guard: we need exactly 3 thresholds to define 4 bands
            if (sorted.Length < 3)
                throw new ArgumentException("T must contain at least 3 threshold values.", nameof(T));

            int[] counts = new int[4];

            foreach (byte p in pixels)
            {
                int k =
                    p <= sorted[0] ? 0 :
                    p <= sorted[1] ? 1 :
                    p <= sorted[2] ? 2 : 3;

                counts[k]++;
            }

            double total = pixels.Length;

            return new SegmentationFeatures
            {
                BackgroundPixels = counts[0],
                LowIntPixels = counts[1],
                MidIntPixels = counts[2],
                TumorPixels = counts[3],
                TotalPixels = pixels.Length,

                BackgroundPct = counts[0] / total * 100,
                LowIntPct = counts[1] / total * 100,
                MidIntPct = counts[2] / total * 100,
                TumorPercentage = counts[3] / total * 100
            };
        }
    }
}