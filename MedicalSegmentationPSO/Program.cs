// Program.cs
using System;
using System.Collections.Generic;
using System.IO;
using BenchmarkDotNet.Running;
using MedicalSegmentationPSO;
using MedicalSegmentationPSO.Benchmark;
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
        BenchmarkPSORunner.Run();

        Console.WriteLine("DONE");
    }
}