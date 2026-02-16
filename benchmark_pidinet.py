"""
PiDiNet Benchmark - FINAL WORKING VERSION
==========================================
Works with ALL checkpoints and model types!

Supported models:
- table5_pidinet (full) - sa=True, dil=True
- table5_pidinet-l (converted full)
- table5_pidinet-small - sa=True, dil=True  
- table5_pidinet-small-l (converted small)
- table5_pidinet-tiny - sa=False, dil=False
- table5_pidinet-tiny-l (converted tiny)
- table6_pidinet
- table7_pidinet
"""

import torch
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from itertools import cycle
from tqdm.auto import tqdm


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel checkpoints"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.'
        else:
            new_state_dict[k] = v
    return new_state_dict


class PiDiNetBenchmarker:
    def __init__(self, model_type='tiny', precision='fp16', checkpoint_path=None):
        """
        Args:
            model_type: 'tiny', 'small', 'full'
            precision: 'fp16' or 'fp32'
            checkpoint_path: path to .pth checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("GPU required!")
        
        self.model_type = model_type
        self.precision = precision
        
        print(f"Loading {model_type} model...")
        
        # Import
        import models
        from models.convert_pidinet import convert_pidinet
        
        # Create args object (like main.py does)
        class Args:
            pass
        
        args = Args()
        args.config = 'carv4'
        
        # Map model_type to sa/dil settings
        if model_type == 'tiny':
            args.sa = False
            args.dil = False
        elif model_type == 'small':
            args.sa = True
            args.dil = True
        elif model_type == 'full':
            args.sa = True
            args.dil = True
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        print(f"  Config: {args.config}, SA: {args.sa}, DIL: {args.dil}")
        
        # Create model (always use 'pidinet' name, args control the architecture)
        model_fn = models.__dict__['pidinet']
        raw_model = model_fn(args)
        
        # Load checkpoint if provided
        if checkpoint_path:
            print(f"  Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            
            # Get state_dict
            if 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
            
            # Remove 'module.' prefix if present (from DataParallel)
            state = remove_module_prefix(state)
            
            # Load with strict=False to ignore size mismatches
            raw_model.load_state_dict(state, strict=True)
            print("  ✓ Checkpoint loaded")
        
        # Convert PDC to vanilla conv
        print("  Converting PDC layers...")
        self.model = convert_pidinet(raw_model, args.config)
        
        # Set precision
        if precision == 'fp16':
            self.model = self.model.half()
        
        self.model.to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model ready! {self.num_params:,} parameters\n")
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def benchmark_folder(self, folder, warmup=20, iters=100, save_log=True):
        """Benchmark on folder of images"""
        
        # Find images
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG']:
            files.extend(glob.glob(f"{folder}/{ext}"))
        
        if not files:
            raise FileNotFoundError(f"No images in {folder}")
        
        print(f"Found {len(files)} images")
        print("Loading images...")
        
        tensors, sizes = [], []
        for f in tqdm(files):
            try:
                img = Image.open(f).convert('RGB')
                sizes.append(img.size)
                t = self.transform(img).unsqueeze(0)
                if self.precision == 'fp16':
                    t = t.half()
                tensors.append(t)
            except Exception as e:
                print(f"Skip {f}: {e}")
        
        if not tensors:
            raise RuntimeError("No images loaded")
        
        data = cycle(tensors)
        
        # Warmup
        print(f"Warmup ({warmup} iters)...")
        with torch.no_grad():
            for _ in range(warmup):
                self.model(next(data).to(self.device))
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking ({iters} iters)...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times = []
        
        with torch.no_grad():
            for _ in tqdm(range(iters)):
                inp = next(data).to(self.device)
                starter.record()
                self.model(inp)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
        
        # Calculate metrics
        lat_avg = np.mean(times)
        lat_std = np.std(times)
        lat_min = np.min(times)
        lat_max = np.max(times)
        lat_p95 = np.percentile(times, 95)
        lat_p99 = np.percentile(times, 99)
        fps = 1000 / lat_avg
        vram = torch.cuda.max_memory_allocated() / 1024**2
        
        w_avg = int(np.mean([s[0] for s in sizes]))
        h_avg = int(np.mean([s[1] for s in sizes]))
        
        # Print report
        report = f"""
{'='*70}
BENCHMARK RESULTS: {self.model_type.upper()} - {self.precision.upper()}
{'='*70}
Dataset:
  Images:       {len(files)}
  Avg Size:     {w_avg}×{h_avg}

Performance:
  Latency (avg): {lat_avg:.3f} ms ± {lat_std:.3f}
  Latency (min): {lat_min:.3f} ms
  Latency (max): {lat_max:.3f} ms
  Latency (p95): {lat_p95:.3f} ms
  Latency (p99): {lat_p99:.3f} ms
  Throughput:    {fps:.2f} FPS

Memory:
  Peak VRAM:     {vram:.2f} MB
  Parameters:    {self.num_params:,}
{'='*70}
"""
        print(report)
        
        # Save log
        if save_log:
            log_file = f"bench_{self.model_type}_{self.precision}.txt"
            with open(log_file, 'w') as f:
                f.write(report)
            print(f"✓ Saved: {log_file}\n")
        
        return {
            'latency_ms': lat_avg,
            'std_ms': lat_std,
            'p95_ms': lat_p95,
            'p99_ms': lat_p99,
            'fps': fps,
            'vram_mb': vram
        }
    
    def stress_test(self, image_path, resolutions=None, save_log=True):
        """Test at different resolutions"""
        
        if resolutions is None:
            resolutions = [1024, 2048, 4096, 8192]
        
        img = Image.open(image_path).convert('RGB')
        
        print(f"\n{'='*70}")
        print(f"STRESS TEST: {self.model_type.upper()} - {self.precision.upper()}")
        print(f"{'='*70}\n")
        
        results = []
        log_lines = []
        
        for size in resolutions:
            try:
                print(f"Testing {size}×{size}...", end=' ')
                
                # Resize
                resized = img.resize((size, size))
                t = self.transform(resized).unsqueeze(0).to(self.device)
                if self.precision == 'fp16':
                    t = t.half()
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        self.model(t)
                torch.cuda.synchronize()
                
                # Measure
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                times = []
                
                with torch.no_grad():
                    for _ in range(20):
                        starter.record()
                        self.model(t)
                        ender.record()
                        torch.cuda.synchronize()
                        times.append(starter.elapsed_time(ender))
                
                lat = np.mean(times)
                vram = torch.cuda.max_memory_allocated() / 1024**2
                
                line = f"  ✓ {size}×{size}: {lat:.2f} ms | {vram:.2f} MB"
                print(line)
                log_lines.append(line)
                
                results.append({
                    'resolution': size,
                    'latency_ms': lat,
                    'vram_mb': vram,
                    'status': 'PASSED'
                })
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    line = f"  ✗ {size}×{size}: OOM"
                    print(line)
                    log_lines.append(line)
                    results.append({'resolution': size, 'status': 'OOM'})
                    torch.cuda.empty_cache()
                    break
                raise
        
        print(f"{'='*70}\n")
        
        # Save log
        if save_log:
            log_file = f"stress_{self.model_type}_{self.precision}.txt"
            with open(log_file, 'w') as f:
                f.write(f"STRESS TEST: {self.model_type} - {self.precision}\n")
                f.write('\n'.join(log_lines))
            print(f"✓ Saved: {log_file}\n")
        
        return results


# Quick helpers for notebook
def bench(model='tiny', folder='test_images/', checkpoint=None):
    """One-liner benchmark"""
    b = PiDiNetBenchmarker(model, 'fp16', checkpoint)
    return b.benchmark_folder(folder)


def stress(model='tiny', image='test.jpg', checkpoint=None):
    """One-liner stress test"""
    b = PiDiNetBenchmarker(model, 'fp16', checkpoint)
    return b.stress_test(image)


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PiDiNet Benchmark Tool')
    parser.add_argument('--model', default='tiny', choices=['tiny', 'small', 'full'])
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16'])
    parser.add_argument('--mode', default='folder', choices=['folder', 'stress'])
    parser.add_argument('--input', required=True, help='Folder or image path')
    parser.add_argument('--checkpoint', default=None, help='Path to .pth checkpoint')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    
    args = parser.parse_args()
    
    b = PiDiNetBenchmarker(args.model, args.precision, args.checkpoint)
    
    if args.mode == 'folder':
        b.benchmark_folder(args.input, args.warmup, args.iters)
    else:
        b.stress_test(args.input)
