// Program.cs
using System;
using System.Collections.Generic;
using System.IO;
using BenchmarkDotNet.Running;
using MedicalSegmentationPSO;
using TorchSharp;
using static TorchSharp.torch;

class Program
{
    const string RawDataset = @"C:\Users\user\BrainTumorDataset";
    const string SegmentedDataset = @"C:\Users\user\SegmentedDataset";
    public const string TestImage = @"C:\MMU\Degree\Sem 5\TPC\MedicalSegmentationPSO\MedicalSegmentationPSO\Tr-no_10.jpg";

    static void Main()
    {
        Console.WriteLine("=== PSO BENCHMARK SYSTEM ===");

        // Load image once
        byte[] pixels = ImageProcessor.LoadGrayscalePixels(
            TestImage,
            out int w,
            out int h);

        // =========================
        // 1. RUN + CACHE SEQUENTIAL PSO
        // =========================
        var seqResult = SequentialPsoRunner.Run(pixels);

        Console.WriteLine("\nSequential PSO (cached or computed)");
        Console.WriteLine($"Time: {seqResult.AverageTimeMs} ms");

        // =========================
        // 2. RUN BENCHMARK DOTNET
        // =========================
        Console.WriteLine("\nRunning BenchmarkDotNet...");

        BenchmarkRunner.Run<PsoBenchmarks>();

        Console.WriteLine("\nDone.");
    }
}