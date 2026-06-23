using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using static Program;

namespace MedicalSegmentationPSO
{
    public static class ResultCache
    {
        private const string Folder = "BenchmarkResults";
        public static void Save(string name, PsoBenchmarkResult result)
        {
            Directory.CreateDirectory(Folder);
            string path = Path.Combine(Folder, $"{name}.json");

            File.WriteAllText(path, JsonSerializer.Serialize(result, new JsonSerializerOptions
            {
                WriteIndented = true
            }));
        }
        
        public static PsoBenchmarkResult Load(string name)
        {
            string path = Path.Combine(Folder, $"{name}.json");

            if (!File.Exists(path))
                return null;

            return JsonSerializer.Deserialize<PsoBenchmarkResult>(File.ReadAllText(path));
        }
    }
}
