using System;
using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace MedicalSegmentationPSO
{
    /// <summary>
    /// Handles loading grayscale pixels and saving colour-mapped segmented images.
    /// </summary>
    public static class ImageProcessor
    {
        // ── Load ────────────────────────────────────────────────────────────
        public static byte[] LoadGrayscalePixels(
            string path, out int width, out int height)
        {
            using Mat img = Cv2.ImRead(path, ImreadModes.Grayscale);

            if (img.Empty())
                throw new Exception($"Could not load image: {path}");

            width = img.Width;
            height = img.Height;

            byte[] pixels = new byte[width * height];
            int idx = 0;

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    pixels[idx++] = img.At<byte>(y, x);

            return pixels;
        }

        // ── Save segmented (grayscale levels) ───────────────────────────────
        /// <summary>
        /// Maps each pixel to one of four intensity bands and writes the result.
        /// Band values: 0 | 85 | 170 | 255 (evenly spaced for clarity).
        /// </summary>
        public static void SaveSegmentedImage(
            byte[] pixels, int width, int height,
            double[] thresholds, string outputPath)
        {
            double[] T = (double[])thresholds.Clone();
            Array.Sort(T);

            using Mat output = new Mat(height, width, MatType.CV_8UC1);

            int idx = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte p = pixels[idx++];

                    byte val =
                        p <= T[0] ? (byte)0 :
                        p <= T[1] ? (byte)85 :
                        p <= T[2] ? (byte)170 :
                                    (byte)255;

                    output.Set(y, x, val);
                }
            }

            Cv2.ImWrite(outputPath, output);
        }

        public static void SaveGrayscale(byte[] pixels, int width, int height, string path)
        {
            using Mat img = new Mat(height, width, MatType.CV_8UC1);

            int idx = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    img.Set(y, x, pixels[idx++]);
                }
            }

            Cv2.ImWrite(path, img);
        }

        // ── Optional preview ────────────────────────────────────────────────
        public static void ShowImage(string windowName, Mat image)
        {
            Cv2.ImShow(windowName, image);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }

        public static Tensor ToTensor(byte[] pixels, int width, int height)
        {
            var gray = torch.tensor(pixels, dtype: ScalarType.Float32)
                .reshape(height, width)
                / 255.0;

            gray = gray.unsqueeze(0);

            gray = gray.repeat(3, 1, 1);

            return gray;
        }
    }
}