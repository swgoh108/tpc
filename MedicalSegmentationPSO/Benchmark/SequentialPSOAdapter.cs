namespace MedicalSegmentationPSO.Benchmark
{
    public class SequentialPSOAdapter : IPSOAlgorithm
    {
        public string Name => "Sequential PSO";

        public double[] Run(byte[] pixels)
        {
            var pso = new PSO(pixels);

            return pso.Run(100);
        }
    }
}