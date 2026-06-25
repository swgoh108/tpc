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
        if (!DatasetSegmentation.IsSegmentedDatasetReady(
                SegmentedDataset))
        {
            Console.WriteLine(
                "\nSegmented dataset not found.");

            DatasetSegmentation.Process(
                RawDataset,
                SegmentedDataset,
                psoIterations: 50);
        }
        else
        {
            Console.WriteLine(
                "\nSegmented dataset already exists.");
        }

        //Console.WriteLine("=== PSO BENCHMARK SYSTEM ===");
        //BenchmarkPSORunner.Run();

        //Console.WriteLine("DONE");

        Console.WriteLine("================================");
        Console.WriteLine(" Brain Tumor Classification");
        Console.WriteLine("================================");

        //--------------------------------
        // TRAINING DATA
        //--------------------------------

        var trainDataset =
            DatasetLoaderTorch.Load(
                @"C:\Users\user\SegmentedDataset\Training");

        //--------------------------------
        // TEST DATA
        //--------------------------------

        var testDataset =
            DatasetLoaderTorch.Load(
                @"C:\Users\user\SegmentedDataset\Testing");

        foreach (var item in trainDataset.Take(1))
        {
            Console.WriteLine(item.Image.shape);
        }

        //--------------------------------
        // TRAIN CNN
        //--------------------------------

        var cnn = new CNNClassifier();

        cnn.Train(
            trainDataset,
            epochs: 10,
            batchSize: 32);

        cnn.Save("BrainTumorCNN.pt");

        //--------------------------------
        // EVALUATE
        //--------------------------------

        var results =
            cnn.Evaluate(testDataset);

        results.Print();

        Console.WriteLine("\nDONE");
    }
}