import cv2
import numpy as np
from GA import GA
import os
import argparse
import matplotlib.pyplot as plt

def mse(original_image, upscaled_image):
    mse = np.mean((original_image.astype(np.float32) - upscaled_image.astype(np.float32)) ** 2)
    return mse

def get_compression_settings(level='medium'):
    """
    Get compression settings based on quality level
    low: fast but lower quality
    medium: balanced
    high: slow but better quality
    """
    settings = {
        'low': {
            'generations': 50,
            'population_size': 50,
            'mutation_rate': 0.01,
            'crossover_rate': 0.8
        },
        'medium': {
            'generations': 100,
            'population_size': 100,
            'mutation_rate': 0.01,
            'crossover_rate': 0.8
        },
        'high': {
            'generations': 200,
            'population_size': 150,
            'mutation_rate': 0.01,
            'crossover_rate': 0.8
        }
    }
    return settings.get(level, settings['medium'])

# def process_image(image_path, compression_level='medium', output_prefix=None):
#     """
#     Process a single image with GA compression
#     Returns: best_individual, fitness_history
#     """
#     print(f"\nProcessing: {image_path}")
#     print(f"Compression level: {compression_level}")
    
#     original = cv2.imread(image_path)
#     if original is None:
#         print(f"Error: Could not load image {image_path}")
#         return None, None
    
#     # Get settings based on compression level
#     settings = get_compression_settings(compression_level)
#     print(f"Settings: {settings}")
    
#     # Generate baseline
#     baseline_bicubic = cv2.resize(original, (200,120), cv2.INTER_CUBIC)
#     baseline_upscaled_bicubic = cv2.resize(baseline_bicubic, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)

#     baseline_bicubic_mse = mse(original, baseline_upscaled_bicubic)

#     baseline_bilinear = cv2.resize(original, (200,120), cv2.INTER_LINEAR)
#     baseline_upscaled_bilinear = cv2.resize(baseline_bilinear, (original.shape[1], original.shape[0]),interpolation=cv2.INTER_LINEAR)
    
#     baseline_bilinear_mse = mse(original, baseline_upscaled_bilinear)

#     print(f"Baseline Bicubic MSE = {baseline_bicubic_mse:.2f}")
#     print(f"Baseline Bilinear MSE = {baseline_bilinear_mse:.2f}")
    
#     # Create GA instance with settings
#     ga = GA(
#         original_image=original,
#         compressed_shape=(120,200,3),
#         population_size=settings['population_size'],
#         crossover_rate=settings['crossover_rate'],
#         mutation_rate=settings['mutation_rate'],
#         crossover='one-point' #write one-point if one point crossover. anything else will do two point crossover
#     )
    
#     # Evolve and get results with fitness history
#     best_individual, fitness_history = ga.evolve(
#         generations=settings['generations'], 
#         baseline=baseline_bicubic
#     )
    
#     # Save compressed images
#     if output_prefix is None:
#         base_name = os.path.splitext(os.path.basename(image_path))[0]
#         output_prefix = f"{base_name}_{compression_level}"
    
#     compressed = best_individual.imgArray
#     compressed_up = cv2.resize(compressed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
#     cv2.imwrite(f"{output_prefix}_compressed.png", compressed)
#     cv2.imwrite(f"{output_prefix}_compressed_upscaled.png", compressed_up)
    
#     final_mse = mse(original, compressed_up)
#     print(f"Final GA MSE = {final_mse:.2f}")
    
#     return best_individual, fitness_history

def process_image_custom(image_path, settings, output_prefix=None):
    """
    Process image with custom settings dictionary
    Returns: best_individual, fitness_history
    """
    print(f"\nProcessing: {image_path}")
    
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return None, None
    
    # Generate baseline
    baseline_bicubic = cv2.resize(original, (200,120), cv2.INTER_CUBIC)
    baseline_upscaled_bicubic = cv2.resize(baseline_bicubic, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)

    baseline_bicubic_mse = mse(original, baseline_upscaled_bicubic)

    baseline_bilinear = cv2.resize(original, (200,120), cv2.INTER_LINEAR)
    baseline_upscaled_bilinear = cv2.resize(baseline_bilinear, (original.shape[1], original.shape[0]),interpolation=cv2.INTER_LINEAR)
    
    baseline_bilinear_mse = mse(original, baseline_upscaled_bilinear)

    print(f"Baseline Bicubic MSE = {baseline_bicubic_mse:.2f}")
    print(f"Baseline Bilinear MSE = {baseline_bilinear_mse:.2f}")

    # Save compressed images
    if output_prefix is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_prefix = base_name

    cv2.imwrite(f"{output_prefix}_baseline_bicubic.png", baseline_bicubic)
    cv2.imwrite(f"{output_prefix}_baseline_bilinear.png", baseline_bilinear)

    # Create GA instance with custom settings
    ga = GA(
        original_image=original,
        compressed_shape=(120,200,3),
        population_size=settings['population_size'],
        crossover_rate=settings['crossover_rate'],
        mutation_rate=settings['mutation_rate'],
        crossover='one_point' #can change this to anything else and it will perform 2 point crossover
    )
    
    # Evolve
    best_individual, fitness_history = ga.evolve(
        generations=settings['generations'],
        baseline=baseline_bicubic
    )
    

    
    compressed = best_individual.imgArray
    compressed_up = cv2.resize(compressed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    cv2.imwrite(f"{output_prefix}_compressed.png", compressed)
    cv2.imwrite(f"{output_prefix}_compressed_upscaled.png", compressed_up)
    
    final_mse = mse(original, compressed_up)
    print(f"Final GA MSE = {final_mse:.2f}")
    
    return best_individual, fitness_history

def print_convergence_summary(fitness_history, image_name="", compression_level=""):
    """
    Print convergence summary (text-based, no matplotlib needed)
    """
    if not fitness_history:
        return
    
    print(f"\n=== Convergence Summary ===")
    if image_name:
        print(f"Image: {image_name}")
    if compression_level:
        print(f"Level: {compression_level}")
    print(f"Initial MSE: {fitness_history[0]:.2f}")
    print(f"Final MSE: {fitness_history[-1]:.2f}")
    print(f"Improvement: {fitness_history[0] - fitness_history[-1]:.2f} ({((fitness_history[0] - fitness_history[-1]) / fitness_history[0] * 100):.1f}%)")
    print(f"Best Generation: {fitness_history.index(min(fitness_history))}")
    print("=" * 30)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Image Compression using Genetic Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python main.py original.jpg
  
  # Multiple images
  python main.py original.jpg test1.jpg test2.jpg
  
  # Use compression level preset
  python main.py original.jpg --level high
  
  # Custom generations and population size
  python main.py original.jpg --generations 200 --population-size 150
  
  # Full custom settings
  python main.py original.jpg -g 300 -p 200 -m 0.05 -c 0.9
  
  # Output to specific directory
  python main.py original.jpg --output-dir results/
        """
    )
    
    # Required arguments
    parser.add_argument('image', nargs='+', 
                       help='Input image file(s) to compress')
    
    # Compression level (preset)
    parser.add_argument('--level', '-l',
                       choices=['low', 'medium', 'high'],
                       default='medium',
                       help='Compression quality level (default: medium)')
    
    # Custom GA parameters (optional, overrides level if provided)
    parser.add_argument('--generations', '-g',
                       type=int,
                       default=None,
                       help='Number of generations (overrides level setting)')
    
    parser.add_argument('--population-size', '-p',
                       type=int,
                       default=None,
                       help='Population size (overrides level setting)')
    
    parser.add_argument('--mutation-rate', '-m',
                       type=float,
                       default=None,
                       help='Mutation rate (0.0-1.0, overrides level setting)')
    
    parser.add_argument('--crossover-rate', '-c',
                       type=float,
                       default=None,
                       help='Crossover rate (0.0-1.0, overrides level setting)')
    
    # Output options
    parser.add_argument('--output-dir', '-o',
                       type=str,
                       default=None,
                       help='Output directory for results (default: current directory)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Get base settings from level
    base_settings = get_compression_settings(args.level)
    
    # Override with custom parameters if provided
    settings = {
        'generations': args.generations if args.generations is not None else base_settings['generations'],
        'population_size': args.population_size if args.population_size is not None else base_settings['population_size'],
        'mutation_rate': args.mutation_rate if args.mutation_rate is not None else base_settings['mutation_rate'],
        'crossover_rate': args.crossover_rate if args.crossover_rate is not None else base_settings['crossover_rate']
    }
    
    print("=== GA Compression Settings ===")
    print(f"Level: {args.level}")
    print(f"Generations: {settings['generations']}")
    print(f"Population Size: {settings['population_size']}")
    print(f"Mutation Rate: {settings['mutation_rate']}")
    print(f"Crossover Rate: {settings['crossover_rate']}")
    print("=" * 30)
    
    # Process each image
    all_results = []
    for img_path in args.image:
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping...")
            continue
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Set output prefix
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_prefix = os.path.join(args.output_dir, f"{base_name}_{args.level}")
        else:
            output_prefix = f"{base_name}_{args.level}"
        
        # Process image with custom settings
        best_individual, fitness_history = process_image_custom(
            img_path,
            settings=settings,
            output_prefix=output_prefix
        )
        
        if best_individual is not None and fitness_history is not None:
            print("\nConvergence Analysis:")
            plt.figure()
            plt.plot(fitness_history)
            plt.xlabel("Generation")
            plt.ylabel("Best MSE")
            plt.title(f"Convergence Analysis for {base_name}")
            plt.grid(True)
            plt.savefig(f"convergence_analysis_{base_name}.png")
            plt.show()
            print_convergence_summary(fitness_history, base_name, args.level)
            all_results.append({
                'image': base_name,
                'fitness_history': fitness_history,
                'final_mse': fitness_history[-1] if fitness_history else None
            })
            
    
    # Print summary
    print("\n=== Summary ===")
    for result in all_results:
        print(f"{result['image']}: Final MSE = {result['final_mse']:.2f}")

if __name__ == "__main__":
    main()
