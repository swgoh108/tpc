using System;
using System.Collections.Generic;
using System.Linq;

namespace MedicalSegmentationPSO
{
    public class EvaluationMetrics
    {
        public double Accuracy;
        public double Precision;
        public double Recall;
        public double F1Score;

        public int[,] ConfusionMatrix;
        public string[] Classes;

        public EvaluationMetrics(int classCount, string[] classes)
        {
            ConfusionMatrix = new int[classCount, classCount];
            Classes = classes;
        }

        // =========================
        // ADD PREDICTION RESULT
        // =========================
        public void Add(int actual, int predicted)
        {
            ConfusionMatrix[actual, predicted]++;
        }

        // =========================
        // COMPUTE METRICS
        // =========================
        public void Compute()
        {
            int classCount = Classes.Length;

            int totalCorrect = 0;
            int totalSamples = 0;

            double precisionSum = 0;
            double recallSum = 0;
            double f1Sum = 0;

            for (int i = 0; i < classCount; i++)
            {
                int tp = ConfusionMatrix[i, i];
                int fp = 0;
                int fn = 0;

                for (int j = 0; j < classCount; j++)
                {
                    if (i != j)
                    {
                        fp += ConfusionMatrix[j, i];
                        fn += ConfusionMatrix[i, j];
                    }
                }

                totalCorrect += tp;

                int rowSum = 0;
                for (int j = 0; j < classCount; j++)
                    rowSum += ConfusionMatrix[i, j];

                totalSamples += rowSum;

                double precision = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
                double recall = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
                double f1 = (precision + recall == 0)
                    ? 0
                    : 2 * precision * recall / (precision + recall);

                precisionSum += precision;
                recallSum += recall;
                f1Sum += f1;
            }

            Accuracy = (double)totalCorrect / totalSamples;
            Precision = precisionSum / classCount;
            Recall = recallSum / classCount;
            F1Score = f1Sum / classCount;
        }

        // =========================
        // PRINT RESULTS
        // =========================
        public void Print()
        {
            Console.WriteLine("\n=== EVALUATION METRICS ===");

            Console.WriteLine($"Accuracy  : {Accuracy:P2}");
            Console.WriteLine($"Precision : {Precision:P2}");
            Console.WriteLine($"Recall    : {Recall:P2}");
            Console.WriteLine($"F1 Score  : {F1Score:P2}");

            Console.WriteLine("\nConfusion Matrix:");

            Console.Write("".PadRight(12));
            foreach (var c in Classes)
                Console.Write($"{c,12}");
            Console.WriteLine();

            for (int i = 0; i < Classes.Length; i++)
            {
                Console.Write($"{Classes[i],12}");

                for (int j = 0; j < Classes.Length; j++)
                    Console.Write($"{ConfusionMatrix[i, j],12}");

                Console.WriteLine();
            }
        }
    }

    public class EvaluationResult
    {
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
    }
}