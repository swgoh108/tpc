namespace MedicalSegmentationPSO.Benchmark
{
    public class SharedMemoryPSOAdapter : IPSOAlgorithm
    {
        public string Name => "Shared Memory PSO";

        public double[] Run(byte[] pixels)
        {
            var pso = new SharedMemoryPSO(
                pixels,
                swarmSize: 30,
                maxThreads: 4);

            return pso.Run(100);
        }
    }
}