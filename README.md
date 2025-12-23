# Image Compression using Genetic Algorithm

An evolutionary approach to image compression that uses genetic algorithms to find optimized low-resolution versions of images, achieving better quality than traditional interpolation methods when upscaled.

## Features

- Compresses images to 120×200×3 pixels using genetic algorithms
- Supports one-point and two-point crossover strategies
- Configurable compression levels (low, medium, high)
- Compares results against baseline bicubic/bilinear interpolation
- Generates convergence analysis plots
- Command-line interface with flexible parameter tuning

## Requirements

- Python 3.x
- numpy
- opencv-python (cv2)
- matplotlib

```bash
pip install numpy opencv-python matplotlib
```

## Usage

### Basic Usage

```bash
python main.py original.jpg
```

### Compression Levels

```bash
# Low quality (fast)
python main.py image.jpg --level low

# Medium quality (default)
python main.py image.jpg --level medium

# High quality (slow)
python main.py image.jpg --level high
```

### Custom Parameters

```bash
python main.py image.jpg --generations 200 --population-size 150 --mutation-rate 0.01 --crossover-rate 0.8
```

### Multiple Images

```bash
python main.py image1.jpg image2.jpg image3.jpg
```

### Output Directory

```bash
python main.py image.jpg --output-dir results/
```

## Output Files

- `*_baseline_bicubic.png` - Baseline compressed image (bicubic)
- `*_baseline_bilinear.png` - Baseline compressed image (bilinear)
- `*_compressed.png` - GA-optimized compressed image (120×200)
- `*_compressed_upscaled.png` - GA-optimized upscaled image
- `convergence_analysis_*.png` - Fitness convergence plot

## Algorithm Overview

1. **Initialization**: Creates population from baseline bicubic-downsampled image with added noise
2. **Fitness Evaluation**: Measures MSE between upscaled candidate and original image
3. **Selection**: Tournament selection for parent selection
4. **Crossover**: One-point or two-point crossover
5. **Mutation**: Random pixel value mutations based on mutation rate
6. **Elitism**: Preserves top 2 individuals to next generation

## Project Structure

- `main.py` - Main entry point and CLI interface
- `GA.py` - Genetic algorithm implementation
- `Individual.py` - Individual candidate solution representation