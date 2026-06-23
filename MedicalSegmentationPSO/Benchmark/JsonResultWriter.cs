using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace MedicalSegmentationPSO.Benchmark
{
    public static class JsonResultWriter
    {
        public static void Save(
            BenchmarkResult result)
        {
            Directory.CreateDirectory("BenchmarkResults");

            string filename =
                $"{result.Algorithm.Replace(" ", "_")}_{DateTime.Now:yyyyMMdd_HHmmss}.json";

            string path =
                Path.Combine(
                    "BenchmarkResults",
                    filename);

            File.WriteAllText(
                path,
                JsonSerializer.Serialize(
                    result,
                    new JsonSerializerOptions
                    {
                        WriteIndented = true
                    }));
        }
    }
}