"""
Simple PiDiNet Benchmarker - FIXED VERSION
Just copy and run this in your notebook or save as .py file
"""

import torch
import torch.nn as nn
import glob
import numpy as np
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from itertools import cycle
from tqdm.auto import tqdm


class PiDiNetBenchmarker:
    def __init__(self, model_type='tiny', precision='fp16', checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("Need GPU!")
        
        self.model_type = model_type
        self.precision = precision
        
        # Import models
        import models
        from models.convert_pidinet import convert_pidinet
        
        # Model configs
        configs = {
            'tiny': {'sa': False, 'dil': False},
            'small': {'sa': True, 'dil': True},
            'full': {'sa': True, 'dil': True},
        }
        cfg = configs[model_type]
        
        print(f"Creating {model_type} model (SA={cfg['sa']}, DIL={cfg['dil']})...")
        
        # Create model - FIXED: Use models.__dict__ not direct import
        pidinet_fn = models.__dict__['pidinet']
        raw_model = pidinet_fn(sa=cfg['sa'], dil=cfg['dil'])
        
        # Load checkpoint
        if checkpoint_path:
            print(f"Loading {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            raw_model.load_state_dict(state)
        
        # Convert PDC layers
        print("Converting model...")
        self.model = convert_pidinet(raw_model, 'carv4')
        
        # Set precision
        if precision == 'fp16':
            self.model = self.model.half()
        
        self.model.to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Ready! {self.num_params:,} parameters\n")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def benchmark_folder(self, folder, warmup=20, iters=100):
        """Benchmark on folder of images"""
        
        # Load images
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            files.extend(glob.glob(f"{folder}/{ext}"))
        
        if not files:
            raise FileNotFoundError(f"No images in {folder}")
        
        print(f"Loading {len(files)} images...")
        tensors = []
        sizes = []
        
        for f in tqdm(files):
            img = Image.open(f).convert('RGB')
            sizes.append(img.size)
            t = self.transform(img).unsqueeze(0)
            if self.precision == 'fp16':
                t = t.half()
            tensors.append(t)
        
        data_pool = cycle(tensors)
        
        # Warmup
        print(f"Warmup ({warmup} iters)...")
        with torch.no_grad():
            for _ in range(warmup):
                self.model(next(data_pool).to(self.device))
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
                inp = next(data_pool).to(self.device)
                starter.record()
                self.model(inp)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
        
        # Stats
        avg_lat = np.mean(times)
        std_lat = np.std(times)
        p95_lat = np.percentile(times, 95)
        fps = 1000 / avg_lat
        vram = torch.cuda.max_memory_allocated() / 1024**2
        
        avg_w = int(np.mean([s[0] for s in sizes]))
        avg_h = int(np.mean([s[1] for s in sizes]))
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.model_type.upper()} - {self.precision.upper()}")
        print(f"{'='*60}")
        print(f"Images:      {len(files)} ({avg_w}×{avg_h} avg)")
        print(f"Latency:     {avg_lat:.2f} ± {std_lat:.2f} ms")
        print(f"P95:         {p95_lat:.2f} ms")
        print(f"FPS:         {fps:.2f}")
        print(f"VRAM:        {vram:.2f} MB")
        print(f"Parameters:  {self.num_params:,}")
        print(f"{'='*60}\n")
        
        return {
            'latency_ms': avg_lat,
            'std_ms': std_lat,
            'p95_ms': p95_lat,
            'fps': fps,
            'vram_mb': vram
        }
    
    def stress_test(self, image_path, resolutions=None):
        """Test at different resolutions"""
        
        if resolutions is None:
            resolutions = [1024, 2048, 4096, 8192]
        
        img = Image.open(image_path).convert('RGB')
        
        print(f"\n{'='*60}")
        print(f"STRESS TEST: {self.model_type.upper()}")
        print(f"{'='*60}\n")
        
        results = []
        
        for size in resolutions:
            try:
                print(f"Testing {size}×{size}...", end=' ')
                
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
                
                print(f"✓ {lat:.2f} ms | {vram:.2f} MB")
                results.append({'resolution': size, 'latency_ms': lat, 'vram_mb': vram})
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("✗ OOM")
                    torch.cuda.empty_cache()
                    break
                raise
        
        print(f"{'='*60}\n")
        return results


# Quick helpers for notebook
def quick_bench(model='tiny', folder='test_images/', checkpoint=None):
    """One-liner benchmark"""
    bench = PiDiNetBenchmarker(model, 'fp16', checkpoint)
    return bench.benchmark_folder(folder)


def quick_stress(model='tiny', image='test.jpg', checkpoint=None):
    """One-liner stress test"""
    bench = PiDiNetBenchmarker(model, 'fp16', checkpoint)
    return bench.stress_test(image)


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tiny', choices=['tiny', 'small', 'full'])
    parser.add_argument('--precision', default='fp16', choices=['fp32', 'fp16'])
    parser.add_argument('--mode', default='folder', choices=['folder', 'stress'])
    parser.add_argument('--input', required=True, help='Folder or image path')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    
    args = parser.parse_args()
    
    bench = PiDiNetBenchmarker(args.model, args.precision, args.checkpoint)
    
    if args.mode == 'folder':
        bench.benchmark_folder(args.input, args.warmup, args.iters)
    else:
        bench.stress_test(args.input)
