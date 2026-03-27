# Digit Recognizer

A cross-platform C++ handwritten digit recognition system. Uses a neural network trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to identify digits (0-9) from images. Optionally accelerates training and inference on AMD GPUs via [ROCm/HIP](https://rocm.docs.amd.com/).

## Features

- **Pure C++17** -- zero external ML framework dependencies; neural network implemented from scratch
- **AMD GPU acceleration** -- optional ROCm/HIP support with console kernel launch logging
- **Cross-platform** -- builds on Windows (MSVC) and Linux (GCC/Clang) with CMake
- **Auto color detection** -- automatically inverts dark-on-light images to match MNIST convention
- **Single & multi-digit** -- recognizes individual digits or sequences of digits in one image
- **Model persistence** -- save/load trained weights in a compact binary format

## Architecture

```
Input Image  ──►  Preprocessing  ──►  Neural Network  ──►  Digit (0-9)
  (any size)      (grayscale,          784 → 256 → 128 → 10
                   invert,             ReLU    ReLU   Softmax
                   center 28×28,
                   normalize)
```

The neural network is a 3-layer MLP (multi-layer perceptron):
- **Input layer**: 784 neurons (28×28 flattened pixels)
- **Hidden layer 1**: 256 neurons, ReLU activation
- **Hidden layer 2**: 128 neurons, ReLU activation
- **Output layer**: 10 neurons, Softmax (one per digit 0-9)

Training uses stochastic gradient descent (SGD) with He weight initialization and cross-entropy loss.

## Project Structure

```
digit-recognizer/
├── CMakeLists.txt              # Cross-platform build (CPU or CPU+GPU)
├── .gitignore
├── README.md
│
├── include/
│   ├── neural_network.h        # MLP with forward/backward pass (CPU + GPU dispatch)
│   ├── gpu_backend.h           # GPU abstraction + kernel launch logger
│   ├── image_processor.h       # Image I/O, grayscale, resize, centering
│   ├── digit_recognizer.h      # High-level recognition API
│   └── mnist_loader.h          # MNIST IDX binary format parser
│
├── src/
│   ├── main.cpp                # CLI entry point
│   ├── neural_network.cpp      # CPU path + GPU dispatch logic
│   ├── gpu_kernels.hip         # 6 HIP kernels (matvec, relu, softmax, backprop, SGD)
│   ├── image_processor.cpp     # stb_image loading, bilinear resize, auto-invert
│   ├── digit_recognizer.cpp    # Train/test/predict orchestration
│   └── mnist_loader.cpp        # Reads MNIST .idx files
│
├── thirdparty/
│   └── stb_image.h             # Public-domain single-header image loader
│
├── scripts/
│   ├── download_mnist.ps1      # Windows PowerShell MNIST downloader
│   └── download_mnist.sh       # Linux/macOS MNIST downloader
│
├── sample_images/              # Example digit images for testing
│   ├── 3.png
│   ├── 5.png
│   ├── 6.png
│   └── 172.png
│
└── data/                       # MNIST files (gitignored; downloaded via scripts)
```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/njobypet/digit-recognizer.git
cd digit-recognizer
```

### 2. Build

**Windows (MSVC)**
```powershell
cmake -B build -S .
cmake --build build --config Release
```

**Linux (GCC/Clang)**
```bash
cmake -B build -S .
cmake --build build
```

**Linux with GPU (ROCm/HIP)**
```bash
cmake -B build -S . -DENABLE_HIP=ON
cmake --build build
```

### 3. Download MNIST

**Windows**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_mnist.ps1
```

**Linux/macOS**
```bash
chmod +x scripts/download_mnist.sh
./scripts/download_mnist.sh
```

### 4. Train

```bash
./digit_recognizer train ./data --epochs 15 --model digit_model.bin
```

### 5. Predict

```bash
./digit_recognizer predict sample_images/5.png --model digit_model.bin
```

## Detailed Usage

### Command Reference

```
digit_recognizer <command> <path> [options]

Commands:
  train          Train a new model on MNIST dataset
  test           Evaluate model accuracy on MNIST test set
  predict        Recognize a single digit from an image file
  predict-multi  Recognize a sequence of digits from an image file

Options:
  --model <file>   Model file path         (default: digit_model.bin)
  --epochs <n>     Number of training passes (default: 10)
  --batch <n>      Mini-batch size          (default: 32)
  --lr <rate>      Learning rate            (default: 0.005)
  --gpu            Enable AMD GPU (ROCm/HIP)
```

### Example: Train a Model

```bash
$ ./digit_recognizer train ./data --epochs 15 --batch 32 --model digit_model.bin

Loading MNIST training data from: ./data
Loaded 60000 images (28x28)
Loaded 60000 labels
Training network...
Architecture: 784 -> 256 -> 128 -> 10
Learning rate: 0.005
Compute device: CPU
Epochs: 15, Batch size: 32
------------------------------------------------------------
Epoch   1/15 | Loss: 0.0587 | Accuracy: 97.03%
Epoch   2/15 | Loss: 0.0145 | Accuracy: 98.24%
Epoch   3/15 | Loss: 0.0081 | Accuracy: 98.78%
Epoch   5/15 | Loss: 0.0034 | Accuracy: 99.40%
Epoch  10/15 | Loss: 0.0009 | Accuracy: 99.90%
Epoch  15/15 | Loss: 0.0003 | Accuracy: 100.00%
Training complete.
Model saved to: digit_model.bin
```

Training typically reaches **97%+ accuracy after 1 epoch** and **99%+ after 5 epochs** on MNIST.

### Example: Predict a Single Digit

```bash
$ ./digit_recognizer predict sample_images/5.png --model digit_model.bin

Model loaded from: digit_model.bin

Prediction Results
------------------------------
Predicted digit: 5
Confidence:      86.3%

All probabilities:
  0: 0.32%
  1: 3.31%
  2: 0.67%
  3: 1.14%
  4: 1.17%
  5: 86.32%  <-- predicted
  6: 1.47%
  7: 0.57%
  8: 2.94%
  9: 2.09%
```

### Example: Predict Multiple Digits

```bash
$ ./digit_recognizer predict-multi sample_images/172.png --model digit_model.bin

Model loaded from: digit_model.bin

Recognized number: 172
```

Multi-digit mode segments the image by detecting vertical whitespace gaps between characters, then recognizes each segment independently.

### Example: Test Against MNIST Test Set

```bash
$ ./digit_recognizer test ./data --model digit_model.bin

Loading MNIST test data from: ./data
Loaded 10000 images (28x28)
Loaded 10000 labels
Test accuracy: 97.85%
```

### Example: GPU-Accelerated Training

```bash
$ ./digit_recognizer train ./data --epochs 15 --gpu --model digit_model.bin

[GPU] AMD GPU detected: AMD Instinct MI210
[GPU] Compute units: 104
[GPU] Global memory: 65520 MB
[GPU] Max threads/block: 1024
[GPU] Neural network GPU acceleration ENABLED
[GPU] Uploaded 3 layers to device memory
------------------------------------------------------------
[GPU 1711582345123ms] Launching kernel: kernel_matvec    grid(1,1,1)  block(256,1,1)
[GPU 1711582345124ms] Launching kernel: kernel_relu      grid(1,1,1)  block(256,1,1)
[GPU 1711582345124ms] Launching kernel: kernel_matvec    grid(1,1,1)  block(256,1,1)
[GPU 1711582345125ms] Launching kernel: kernel_relu      grid(1,1,1)  block(256,1,1)
[GPU 1711582345125ms] Launching kernel: kernel_matvec    grid(1,1,1)  block(256,1,1)
[GPU 1711582345126ms] Launching kernel: kernel_softmax   grid(1,1,1)  block(32,1,1)  shared=256B
[GPU 1711582345127ms] Launching kernel: kernel_output_delta     grid(1,1,1)  block(256,1,1)
[GPU 1711582345128ms] Launching kernel: kernel_backprop_delta   grid(1,1,1)  block(256,1,1)
[GPU 1711582345129ms] Launching kernel: kernel_update_weights   grid(1,1,1)  block(256,1,1)
Epoch   1/15 | Loss: 0.0587 | Accuracy: 97.03%
...
```

Kernel logging is throttled to the first batch of each epoch to keep output readable.

## GPU Kernels

When built with `-DENABLE_HIP=ON`, six HIP kernels run on the AMD GPU:

| Kernel | Operation | Phase |
|--------|-----------|-------|
| `kernel_matvec` | W × x + b (matrix-vector multiply) | Forward |
| `kernel_relu` | max(0, x) element-wise | Forward |
| `kernel_softmax` | Parallel softmax with shared-memory reduction | Forward |
| `kernel_output_delta` | output - target (cross-entropy gradient) | Backward |
| `kernel_backprop_delta` | Hidden layer gradient propagation | Backward |
| `kernel_update_weights` | w -= lr × δ × a (SGD update) | Update |

Every kernel launch prints a console log line with the kernel name, grid dimensions, block dimensions, and shared memory size.

If no AMD GPU is available, `--gpu` falls back gracefully:
```
[GPU] ROCm/HIP not available (built without USE_HIP). Using CPU.
```

## Image Preprocessing Pipeline

The preprocessing pipeline converts any input image into the 784-element vector the network expects:

1. **Load** -- stb_image reads PNG, JPEG, BMP, GIF, TGA, PSD, HDR, or PIC
2. **Grayscale** -- weighted RGB-to-luminance conversion (0.299R + 0.587G + 0.114B)
3. **Auto-invert** -- if the image border is mostly bright (dark digit on white paper), the pixels are inverted to match MNIST's white-on-black convention
4. **Bounding box crop** -- finds the tightest rectangle around non-zero pixels
5. **Aspect-preserving resize** -- scales into a 20×20 box using bilinear interpolation
6. **Center** -- places the 20×20 digit in the middle of a 28×28 canvas
7. **Normalize** -- scales pixel values from [0, 255] to [0.0, 1.0]

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| CMake | 3.14+ | Build system |
| C++ compiler | C++17 | MSVC 2019+, GCC 7+, or Clang 5+ |
| ROCm | 5.0+ | Optional, for GPU acceleration |

## Tips for Best Results

- **Contrast**: use thick strokes on a clean background
- **Centering**: the digit should roughly fill the image
- **Color**: both dark-on-light and light-on-dark are supported (auto-detected)
- **Multi-digit**: leave clear horizontal gaps between digits
- **Format**: PNG recommended for lossless quality

## License

`stb_image.h` is public domain (Sean Barrett). The rest of the project code is provided as-is for educational and practical use.
