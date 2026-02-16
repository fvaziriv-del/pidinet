"""
Fixed PiDiNet Benchmarker for Jupyter Notebook
Corrected issues:
1. Import path for convert_pidinet (it's in models/convert_pidinet, not utils)
2. Model architecture naming (tiny uses 'carv4', not 'ant')
3. Added better error handling
4. Notebook-friendly output
"""

import sys
import os
import torch
import torch.nn as nn
import time
import glob
import platform
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from itertools import cycle

# Repository imports - CORRECTED PATHS
try:
    # Import from models directory
    from models.convert_pidinet import convert_pidinet
    from models.pidinet import pidinet
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nMake sure you're in the pidinet repository root directory")
    print("Repository structure should be:")
    print("  pidinet/")
    print("    ├── models/")
    print("    │   ├── __init__.py")
    print("    │   ├── pidinet.py")
    print("    │   ├── convert_pidinet.py")
    print("    │   └── config.py")
    print("    ├── utils.py")
    print("    └── benchmark_pidinet.py (this file)")
    raise


class PiDiNetBenchmarker:
    """
    Benchmark tool for PiDiNet edge detection models
    
    Args:
        model_type: 'tiny', 'small', or 'full' 
        precision: 'fp32' or 'fp16'
        use_converted: Whether to convert PDC layers to vanilla conv
        checkpoint_path: Path to .pth checkpoint file
    """
    
    def __init__(self, model_type='tiny', precision='fp32', use_converted=True, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type != 'cuda':
            raise RuntimeError("Benchmark requires an NVIDIA GPU with CUDA support.")
        
        self.model_type = model_type
        self.precision = precision
        self.use_converted = use_converted
        
        # CORRECTED: Model configuration mapping
        # Based on repository scripts.sh and README
        model_configs = {
            'tiny': {'config': 'carv4', 'sa': False, 'dil': False},      # pidinet-tiny
            'small': {'config': 'carv4', 'sa': True, 'dil': True},       # pidinet-small
            'full': {'config': 'carv4', 'sa': True, 'dil': True},        # pidinet (full)
        }
        
        if model_type not in model_configs:
            raise ValueError(f"model_type must be one of {list(model_configs.keys())}")
        
        config_params = model_configs[model_type]
        
        # Create base model
        print(f"Creating {model_type} PiDiNet model...")
        print(f"  Config: {config_params['config']}, SA: {config_params['sa']}, DIL: {config_params['dil']}")
        
        raw_model = pidinet(
            config=config_params['config'],
            sa=config_params['sa'],
            dil=config_params['dil']
        )
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            raw_model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully")
        elif checkpoint_path:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
        # Convert model if requested
        if use_converted:
            print("Converting PDC layers to vanilla convolutions...")
            self.model = convert_pidinet(raw_model, config_params['config'])
        else:
            self.model = raw_model
        
        # Set precision
        if self.precision == 'fp16':
            print("Converting model to FP16...")
            self.model = self.model.half()
        
        # Move to device and set to eval mode
        self.model.to(self.device).eval()
        
        # Calculate parameters
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model ready: {self.total_params:,} parameters")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def run_folder_benchmark(self, folder_path, warmup_iters=20, test_iters=100, save_log=True):
        """
        Benchmark on a folder of images
        
        Args:
            folder_path: Directory containing test images
            warmup_iters: Number of warmup iterations
            test_iters: Number of timed iterations
            save_log: Whether to save results to file
        """
        # Find images
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder_path, e)))
        
        if not files:
            raise FileNotFoundError(f"No images found in {folder_path}")
        
        print(f"Found {len(files)} images in {folder_path}")
        
        # Load and preprocess images
        tensors = []
        res_list = []
        
        for f in files:
            try:
                img = Image.open(f).convert('RGB')
                res_list.append(img.size)
                t = self.transform(img).unsqueeze(0)
                
                if self.precision == 'fp16':
                    t = t.half()
                
                tensors.append(t)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        
        if not tensors:
            raise RuntimeError("No images could be loaded successfully")
        
        print(f"Successfully loaded {len(tensors)} images")
        
        # Create cycling iterator
        data_pool = cycle(tensors)
        
        # Warmup
        print(f"Running {warmup_iters} warmup iterations...")
        with torch.no_grad():
            for i in range(warmup_iters):
                _ = self.model(next(data_pool).to(self.device))
                if i % 10 == 0:
                    print(f"  Warmup: {i}/{warmup_iters}")
        
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Running {test_iters} benchmark iterations...")
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_init = torch.cuda.memory_allocated() / 1024**2
        
        with torch.no_grad():
            for i in range(test_iters):
                inp = next(data_pool).to(self.device)
                starter.record()
                _ = self.model(inp)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
                
                if (i + 1) % 20 == 0:
                    print(f"  Benchmark: {i+1}/{test_iters}")
        
        # Calculate statistics
        avg_lat = np.mean(timings)
        std_lat = np.std(timings)
        min_lat = np.min(timings)
        max_lat = np.max(timings)
        fps = 1000 / avg_lat
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        
        avg_w = int(np.mean([r[0] for r in res_list]))
        avg_h = int(np.mean([r[1] for r in res_list]))
        
        # Generate report
        report = (
            f"{'='*70}\n"
            f"FOLDER BENCHMARK RESULTS\n"
            f"{'='*70}\n"
            f"Model: {self.model_type.upper()} | Precision: {self.precision.upper()} | "
            f"Converted: {self.use_converted}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*70}\n"
            f"Dataset:\n"
            f"  Images: {len(files)}\n"
            f"  Avg Resolution: {avg_w}×{avg_h}\n"
            f"{'='*70}\n"
            f"Performance:\n"
            f"  Latency (avg): {avg_lat:.3f} ms (±{std_lat:.3f})\n"
            f"  Latency (min): {min_lat:.3f} ms\n"
            f"  Latency (max): {max_lat:.3f} ms\n"
            f"  Throughput: {fps:.2f} FPS\n"
            f"{'='*70}\n"
            f"Memory:\n"
            f"  Peak VRAM: {peak_vram:.2f} MB\n"
            f"  Parameters: {self.total_params:,}\n"
            f"{'='*70}\n"
        )
        
        print(report)
        
        if save_log:
            log_name = f"bench_folder_{self.model_type}_{self.precision}.txt"
            with open(log_name, 'w') as f:
                f.write(report)
            print(f"✓ Log saved: {log_name}")
        
        return {
            'avg_latency': avg_lat,
            'std_latency': std_lat,
            'fps': fps,
            'peak_vram_mb': peak_vram,
            'num_images': len(files),
            'avg_resolution': (avg_w, avg_h)
        }
    
    def run_stress_test(self, image_path, resolutions=None, save_log=True):
        """
        Stress test at different resolutions
        
        Args:
            image_path: Path to a single test image
            resolutions: List of square resolutions to test (default: [1024, 2048, 4096, 8192])
            save_log: Whether to save results to file
        """
        if resolutions is None:
            resolutions = [1024, 2048, 4096, 8192]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        
        print(f"\n{'='*70}")
        print(f"STRESS TEST: {self.model_type.upper()} | {self.precision.upper()}")
        print(f"{'='*70}\n")
        
        log_entries = []
        results = []
        
        for side in resolutions:
            try:
                print(f"Testing resolution: {side}×{side}...")
                
                # Resize image
                img_resized = img.resize((side, side))
                t = self.transform(img_resized).unsqueeze(0).to(self.device)
                
                if self.precision == 'fp16':
                    t = t.half()
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = self.model(t)
                    torch.cuda.synchronize()
                
                # Benchmark
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                latencies = []
                with torch.no_grad():
                    for _ in range(20):
                        starter.record()
                        _ = self.model(t)
                        ender.record()
                        torch.cuda.synchronize()
                        latencies.append(starter.elapsed_time(ender))
                
                vram = torch.cuda.max_memory_allocated() / 1024**2
                avg_lat = np.mean(latencies)
                
                status = f"  ✓ {side}×{side} | Latency: {avg_lat:.2f} ms | VRAM: {vram:.2f} MB"
                print(status)
                
                log_entries.append(status)
                results.append({
                    'resolution': side,
                    'latency_ms': avg_lat,
                    'vram_mb': vram,
                    'status': 'PASSED'
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    status = f"  ✗ {side}×{side} | FAILED (Out of Memory)"
                    print(status)
                    log_entries.append(status)
                    results.append({
                        'resolution': side,
                        'status': 'OOM'
                    })
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
        
        print(f"\n{'='*70}\n")
        
        if save_log:
            log_name = f"stress_{self.model_type}_{self.precision}.txt"
            with open(log_name, 'w') as f:
                f.write(f"STRESS TEST RESULTS\n")
                f.write(f"Model: {self.model_type} | Precision: {self.precision}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("\n".join(log_entries))
            print(f"✓ Log saved: {log_name}")
        
        return results


# Notebook-friendly function
def create_benchmarker(model_type='tiny', precision='fp16', checkpoint_path=None):
    """
    Convenience function for notebook use
    
    Example:
        bench = create_benchmarker('tiny', 'fp16', 'trained_models/table5_pidinet.pth')
        results = bench.run_folder_benchmark('test_images/')
    """
    return PiDiNetBenchmarker(
        model_type=model_type,
        precision=precision,
        use_converted=True,
        checkpoint_path=checkpoint_path
    )


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PiDiNet Benchmarking Tool')
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small', 'full'],
                       help='Model size: tiny, small, or full')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16'],
                       help='Precision: fp32 or fp16')
    parser.add_argument('--mode', type=str, default='stress', choices=['stress', 'folder'],
                       help='Benchmark mode')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image (stress mode) or folder (folder mode)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (.pth)')
    parser.add_argument('--warmup', type=int, default=20,
                       help='Warmup iterations for folder mode')
    parser.add_argument('--iters', type=int, default=100,
                       help='Test iterations for folder mode')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Create benchmarker
    bench = PiDiNetBenchmarker(
        model_type=args.model,
        precision=args.precision,
        use_converted=True,
        checkpoint_path=args.checkpoint
    )
    
    # Run benchmark
    if args.mode == 'stress':
        bench.run_stress_test(args.input, save_log=args.save)
    else:
        bench.run_folder_benchmark(args.input, args.warmup, args.iters, save_log=args.save)
