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
├── sample_images/              # 100 MNIST samples + hand-drawn test images
│   ├── digit_0_0.bmp … digit_9_9.bmp   (10 per digit, from MNIST)
│   ├── 3.png, 5.png, 6.png, 172.png    (hand-drawn samples)
│
├── test/
│   ├── stress_test.cpp         # External stress test (spawns predict per image)
│   └── stop_digit_recognizer.cpp  # Signals --infinite loop to exit gracefully
│
├── tools/
│   └── generate_samples.cpp    # Extracts MNIST images to BMP files
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
  predict        Recognize a single digit from an image file (or directory with --infinite)
  predict-multi  Recognize a sequence of digits from an image file

Options:
  --model <file>     Model file path              (default: digit_model.bin)
  --epochs <n>       Number of training passes    (default: 10)
  --batch <n>        Mini-batch size              (default: 32)
  --lr <rate>        Learning rate                (default: 0.005)
  --gpu              Enable AMD GPU (ROCm/HIP)
  --verbose          Enable all diagnostic logs (CPU + GPU)
  --cpulogs on|off   Turn CPU-side logs on or off
  --gpulogs on|off   Turn GPU-side logs on or off
  --infinite         Run predict in an infinite loop on random images from a directory
  --gpudelay         Inject random 2-10s delays into ~10% of GPU kernels
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

## Diagnostic Logging

Three flags control the verbosity of console logs during `predict` and `predict-multi`:

| Flag | Description |
|------|-------------|
| `--cpulogs on` | Show CPU-side logs: image loading, memory allocation, preprocessing, CPU inference steps |
| `--cpulogs off` | Suppress all CPU-side logs |
| `--gpulogs on` | Show GPU-side logs: `hipMalloc`/`hipFree`, host↔device data copies, kernel launch and completion |
| `--gpulogs off` | Suppress all GPU-side logs |
| `--verbose` | Shorthand for `--cpulogs on --gpulogs on` |

When both `--verbose` and an explicit `--cpulogs`/`--gpulogs` flag are given, the explicit flag takes precedence. This lets you use `--verbose` as a baseline and selectively silence one side.

### Example: CPU logs only

```bash
$ ./digit_recognizer predict sample_images/5.png --model digit_model.bin --cpulogs on

Model loaded from: digit_model.bin
[CPU 35620065ms] === Loading and preprocessing image ===
[CPU 35620065ms] Image path: sample_images/5.png
[CPU 35620065ms] Loading image from disk...
[CPU 35620067ms] Loaded: 1152x720 pixels, 4 channels
[CPU 35620067ms] Raw image pixel buffer allocated on CPU  size=3317760 bytes
[CPU 35620067ms] Preprocessing: grayscale, auto-invert, center, normalize...
[CPU 35620070ms] Preprocessed input vector (28x28 = 784 doubles) on CPU  size=6272 bytes
[CPU 35620070ms] === Preprocessing complete ===
[LOG 35620070ms] === Running neural network inference ===
[LOG 35620070ms] Compute path: CPU
[CPU 35620070ms] --- CPU predict: begin forward pass ---
[CPU 35620070ms] Input vector allocated on CPU  size=6272 bytes
[CPU 35620070ms]   Layer 0 (784 -> 256): computing matvec on CPU...
[CPU 35620070ms]   Layer 0 (784 -> 256): output buffer allocated on CPU  size=2048 bytes
[CPU 35620070ms]   Layer 0 (784 -> 256): applied ReLU activation
[CPU 35620070ms]   Layer 1 (256 -> 128): computing matvec on CPU...
[CPU 35620070ms]   Layer 1 (256 -> 128): output buffer allocated on CPU  size=1024 bytes
[CPU 35620070ms]   Layer 1 (256 -> 128): applied ReLU activation
[CPU 35620070ms]   Layer 2 (128 -> 10): computing matvec on CPU...
[CPU 35620070ms]   Layer 2 (128 -> 10): output buffer allocated on CPU  size=80 bytes
[CPU 35620070ms]   Output layer: computing softmax on CPU...
[CPU 35620070ms] --- CPU predict: forward pass complete ---
[LOG 35620070ms] === Inference complete: digit=5 confidence=87.53% ===

Prediction Results
------------------------------
Predicted digit: 5
Confidence:      87.5%
```

### Example: GPU logs only

```bash
$ ./digit_recognizer predict sample_images/5.png --model digit_model.bin --gpu --gpulogs on

Model loaded from: digit_model.bin
[GPU 12345678ms] --- GPU predict: begin forward pass ---
[GPU 12345678ms] Allocating GPU memory for input vector...
[GPU 12345678ms] hipMalloc (GPU memory allocated)  size=6272 bytes  ptr=0x7f...
[GPU 12345679ms] Copying input data from CPU to GPU...
[MEM 12345679ms] Copy HOST -> DEVICE (CPU to GPU)  size=6272 bytes
[GPU 12345679ms]   Layer 0 (784 -> 256): allocating output buffer on GPU...
[GPU 12345679ms] hipMalloc (GPU memory allocated)  size=2048 bytes  ptr=0x7f...
[GPU 12345680ms]   Layer 0 (784 -> 256): launching matvec kernel...
[GPU 12345680ms] Launching kernel: kernel_matvec  grid(1,1,1)  block(256,1,1)
[GPU 12345680ms] Kernel complete: kernel_matvec
[GPU 12345680ms]   Layer 0 (784 -> 256): launching ReLU kernel...
[GPU 12345680ms] Launching kernel: kernel_relu  grid(1,1,1)  block(256,1,1)
[GPU 12345681ms] Kernel complete: kernel_relu
...
[GPU 12345682ms] Synchronizing GPU (waiting for all kernels to complete)...
[GPU 12345682ms] hipDeviceSynchronize -- waiting for all GPU kernels to finish...
[GPU 12345682ms] hipDeviceSynchronize -- complete
[GPU 12345683ms] Copying result data from GPU to CPU...
[MEM 12345683ms] Copy DEVICE -> HOST (GPU to CPU)  size=80 bytes
[GPU 12345683ms] Freeing temporary GPU memory...
[GPU 12345683ms] hipFree (GPU memory freed)  ptr=0x7f...
[GPU 12345684ms] --- GPU predict: forward pass complete ---

Prediction Results
------------------------------
Predicted digit: 5
Confidence:      87.5%
```

### Example: All logs (verbose)

```bash
$ ./digit_recognizer predict sample_images/5.png --model digit_model.bin --gpu --verbose
```

This prints both CPU and GPU logs interleaved in execution order.

### Example: Verbose with CPU logs suppressed

```bash
$ ./digit_recognizer predict sample_images/5.png --model digit_model.bin --gpu --verbose --cpulogs off
```

`--verbose` enables both, then `--cpulogs off` overrides just the CPU side, leaving only GPU logs.

### Log Tag Reference

| Tag | Source | Meaning |
|-----|--------|---------|
| `[CPU ...]` | `--cpulogs` | CPU memory allocation, preprocessing, CPU inference |
| `[GPU ...]` | `--gpulogs` | GPU memory alloc/free, kernel launch, kernel complete |
| `[MEM ...]` | `--gpulogs` | Data transfers: host→device, device→host, device→device |
| `[LOG ...]` | either | Shared phase markers (shown if either flag is on) |

## Infinite Predict Mode

The `--infinite` flag runs `predict` in a continuous loop, picking random images from a directory. This is useful for GPU burn-in testing, profiling, or long-running demo scenarios.

### Starting the loop

Pass a **directory** (not a single file) as the path argument:

```bash
$ ./digit_recognizer predict sample_images --model digit_model.bin --infinite --gpu

=== Infinite predict mode ===
Images dir:  sample_images
Image count: 104
GPU:         yes
PID file:    .digit_recognizer.pid
Stop with:   stop_digit_recognizer  or  Ctrl+C

[     1] digit=1  conf=97.2%  file=digit_1_5.bmp
[     2] digit=8  conf=100.0%  file=digit_8_3.bmp
[     3] digit=4  conf=100.0%  file=digit_4_4.bmp
[     4] digit=2  conf=100.0%  file=digit_2_9.bmp
...
```

All standard flags work with `--infinite`:

```bash
# Infinite with GPU logs
./digit_recognizer predict sample_images --model m.bin --infinite --gpu --gpulogs on

# Infinite with all logs
./digit_recognizer predict sample_images --model m.bin --infinite --verbose

# Infinite CPU-only
./digit_recognizer predict sample_images --model m.bin --infinite
```

### Stopping the loop

**Option 1: `stop_digit_recognizer`** (from another terminal)

```bash
$ ./stop_digit_recognizer

Sending stop signal to digit_recognizer (PID 12345)...
digit_recognizer stopped successfully.
```

`stop_digit_recognizer` creates a signal file (`.digit_recognizer.stop`), waits for the loop to exit, and cleans up. The loop finishes its current prediction and exits gracefully with a summary.

**Option 2: Ctrl+C** in the running terminal

Both methods produce a summary on exit:

```
=== Stopped ===
Reason: stop_digit_recognizer signal received
Total predictions: 20819
Elapsed: 23.0s
Avg per prediction: 1ms
```

### How it works

The infinite loop uses cross-platform file-based signaling:

1. `digit_recognizer --infinite` writes its PID to `.digit_recognizer.pid`
2. Each iteration, it checks if `.digit_recognizer.stop` exists
3. `stop_digit_recognizer` creates `.digit_recognizer.stop` and waits up to 30 seconds for the PID file to disappear
4. On exit, `digit_recognizer` removes both `.pid` and `.stop` files

No platform-specific IPC is used -- works identically on Windows and Linux.

## GPU Kernel Delay Injection (`--gpudelay`)

The `--gpudelay` flag injects random delays (2-10 seconds) into approximately 10% of GPU kernel launches. This is useful for:

- **Profiling tool testing** -- simulates slow kernels to verify profiler behavior
- **Timeout/watchdog testing** -- ensures your monitoring detects stalled GPU work
- **Stress testing** -- creates uneven GPU utilization patterns

When a kernel is selected for delay, its name is prefixed with `delay_` in all log output so it's easy to identify.

### Usage

```bash
# Single predict with delay injection and GPU logs
./digit_recognizer predict sample_images/digit_5_0.bmp --model m.bin --gpu --gpudelay --gpulogs on

# Infinite loop with delays
./digit_recognizer predict sample_images --model m.bin --gpu --infinite --gpudelay --gpulogs on

# Stress test with delays
./stress_test --exe ./digit_recognizer --model m.bin --images sample_images --gpu --gpudelay --gpulogs on
```

### Example output

```
[GPU 12345678ms] Launching kernel: kernel_matvec  grid(1,1,1)  block(256,1,1)
[GPU 12345678ms] Kernel complete: kernel_matvec
[GPU 12345679ms] Launching kernel: kernel_relu  grid(1,1,1)  block(256,1,1)
[GPU 12345679ms] Kernel complete: kernel_relu
[DELAY 12345680ms] Injecting 7234ms delay before delay_kernel_matvec
[GPU 12352914ms] Launching kernel: delay_kernel_matvec  grid(1,1,1)  block(256,1,1)
[GPU 12352914ms] Kernel complete: delay_kernel_matvec
[GPU 12352915ms] Launching kernel: kernel_relu  grid(1,1,1)  block(256,1,1)
[GPU 12352915ms] Kernel complete: kernel_relu
[GPU 12352916ms] Launching kernel: kernel_softmax  grid(1,1,1)  block(32,1,1)  shared=256B
[GPU 12352916ms] Kernel complete: kernel_softmax
```

### Behavior

| Aspect | Detail |
|--------|--------|
| Probability | ~10% of kernel launches are delayed |
| Delay range | 2,000ms to 10,000ms (uniform random) |
| Naming | Delayed kernels: `delay_kernel_matvec`, `delay_kernel_relu`, etc. |
| Non-delayed | Normal kernel name: `kernel_matvec`, `kernel_relu`, etc. |
| Scope | Applies to all 6 GPU kernels equally |
| Log tag | `[DELAY ...]` line printed before the sleep |

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
