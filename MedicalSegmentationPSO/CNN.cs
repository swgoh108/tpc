using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace MedicalSegmentationPSO
{
    // =========================
    // CNN MODEL
    // =========================
    public class BrainTumorCNN : Module<Tensor, Tensor>
    {
        // FIX 1: Use concrete TorchSharp module types instead of base Module
        //        so that .forward() resolves correctly on each layer.
        private readonly Conv2d conv1;
        private readonly Conv2d conv2;
        private readonly Conv2d conv3;
        private readonly Linear fc1;
        private readonly Linear fc2;
        private bool fcInitialized = false;

        // Input assumed to be 224x224. After 3x MaxPool2d(2): 224→112→56→28
        // Flattened size = 128 * 28 * 28 = 100352
        private const int FlatSize = 128 * 28 * 28;

        public BrainTumorCNN() : base("BrainTumorCNN")
        {
            conv1 = Conv2d(3, 32, 3, padding: 1);
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

            // FIX: normalize spatial size
            x = functional.adaptive_avg_pool2d(x, new long[] { 7, 7 });

            x = x.flatten(1);

            x = functional.relu(fc1.forward(x));
            x = fc2.forward(x);

            return x;
        }
    }

    // =========================
    // TRAINER
    // =========================
    public class CNNClassifier
    {
        private readonly BrainTumorCNN model;
        private readonly Device device;

        public CNNClassifier()
        {
            device = cuda.is_available() ? CUDA : CPU;
            model = new BrainTumorCNN();
            model.to(device);
        }

        // =========================
        // TRAIN
        // =========================
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
                    $"Epoch {epoch + 1}/{epochs} | Loss={epochLoss:F4}");
            }

            Console.WriteLine("Training Completed.");
        }

        // =========================
        // PREDICT
        // =========================
        public int Predict(Tensor image)
        {
            model.eval();

            using var noGrad = torch.no_grad();

            var output = model.forward(image.to(device));
            var pred = output.argmax(1);

            return (int)pred.item<long>();
        }

        // =========================
        // SAVE
        // =========================
        public void Save(string path)
        {
            // FIX 4: TorchSharp uses model.save(path), not torch.save(state_dict, path)
            model.save(path);
            Console.WriteLine($"Model saved → {path}");
        }

        // =========================
        // LOAD
        // =========================
        public void Load(string path)
        {
            // FIX 4: TorchSharp uses model.load(path), not torch.load(path)
            model.load(path);
            model.eval();
            Console.WriteLine($"Model loaded ← {path}");
        }
    }

    // =========================
    // LABEL MAP
    // =========================
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