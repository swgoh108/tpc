namespace MedicalSegmentationPSO.Benchmark
{
    public class MultiCorePSOAdapter : IPSOAlgorithm
    {
        public string Name => "Multi-Core PSO";

        public double[] Run(byte[] pixels)
        {
            var pso = new MultiCorePSO(pixels);

            return pso.Run(100);
        }
    }
}