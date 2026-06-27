//CNN.cs
using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MedicalSegmentationPSO
{
    public class BrainTumorCNN : Module<Tensor, Tensor>
    {
        private readonly Conv2d conv1;
        private readonly Conv2d conv2;
        private readonly Conv2d conv3;

        private readonly Linear fc1;
        private readonly Linear fc2;

        private const int FlatSize = 128 * 7 * 7;

        public BrainTumorCNN() : base("BrainTumorCNN")
        {
            // MRI images are grayscale
            conv1 = Conv2d(1, 32, 3, padding: 1);

            conv2 = Conv2d(32, 64, 3, padding: 1);

            conv3 = Conv2d(64, 128, 3, padding: 1);

            fc1 = Linear(FlatSize, 512);

            fc2 = Linear(512, 4);

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            x = functional.relu(conv1.forward(x));
            x = functional.max_pool2d(x, 2);

            x = functional.relu(conv2.forward(x));
            x = functional.max_pool2d(x, 2);

            x = functional.relu(conv3.forward(x));
            x = functional.max_pool2d(x, 2);

            x = functional.adaptive_avg_pool2d(
                x,
                new long[] { 7, 7 });

            x = x.flatten(1);

            x = functional.relu(fc1.forward(x));

            x = fc2.forward(x);

            return x;
        }
    }

    public class CNNClassifier
    {
        private readonly BrainTumorCNN model;

        private readonly Device device;

        public CNNClassifier()
        {
            device =
                cuda.is_available()
                ? CUDA
                : CPU;

            model = new BrainTumorCNN();

            model.to(device);
        }

        public void Train(
            List<(Tensor Image, Tensor Label)> dataset,
            int epochs = 10,
            int batchSize = 32)
        {
            var criterion = CrossEntropyLoss();

            var optimizer =
                optim.Adam(
                    model.parameters(),
                    lr: 0.001);

            Console.WriteLine("CNN Training Started...");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                model.train();

                float epochLoss = 0;

                foreach (var batch in TorchDataLoader.CreateBatches(
                             dataset,
                             batchSize))
                {
                    optimizer.zero_grad();

                    var images =
                        batch.Images.to(device);

                    var labels =
                        batch.Labels.to(device);

                    var outputs =
                        model.forward(images);

                    var loss =
                        criterion.forward(
                            outputs,
                            labels);

                    loss.backward();

                    optimizer.step();

                    epochLoss += loss.item<float>();
                }

                Console.WriteLine(
                    $"Epoch {epoch + 1}/{epochs} | Loss = {epochLoss:F4}");
            }

            Console.WriteLine("Training Completed.");
        }

        public int Predict(Tensor image)
        {
            model.eval();

            using var noGrad = torch.no_grad();

            var output =
                model.forward(
                    image.to(device));

            var prediction =
                output.argmax(1);

            return (int)prediction.item<long>();
        }

        public EvaluationMetrics Evaluate(
            List<(Tensor Image, Tensor Label)> testDataset)
        {
            model.eval();

            var metrics =
                new EvaluationMetrics(
                    TumorLabels.Classes.Length,
                    TumorLabels.Classes);

            using var noGrad = torch.no_grad();

            foreach (var sample in testDataset)
            {
                int actual =
                    (int)sample.Label.item<long>();

                int predicted =
                    Predict(sample.Image.unsqueeze(0));

                metrics.Add(
                    actual,
                    predicted);
            }

            metrics.Compute();

            return metrics;
        }

        public void Save(string path)
        {
            model.save(path);
        }

        public void Load(string path)
        {
            model.load(path);
            model.eval();
        }
    }

    public static class TumorLabels
    {
        public static readonly string[] Classes =
        {
            "glioma",
            "meningioma",
            "pituitary",
            "notumor"
        };
    }
}