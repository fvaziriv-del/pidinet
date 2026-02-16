"""
PiDiNet Benchmark - Based on throughput.py from the repo
=========================================================
No conversion - direct benchmarking like the official throughput.py
"""

import torch
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from itertools import cycle
from tqdm.auto import tqdm


def remove_module_prefix(state_dict):
    """Remove module. prefix from DataParallel"""
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}


class PiDiNetBenchmarker:
    def __init__(self, model_type='tiny', precision='fp16', checkpoint_path=None):
        """
        Args:
            model_type: 'tiny' (no sa/dil), 'small' (sa/dil), 'full' (sa/dil)
            precision: 'fp16' or 'fp32'
            checkpoint_path: .pth file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("GPU required!")
        
        self.model_type = model_type
        self.precision = precision
        
        print(f"Creating {model_type} model...")
        
        # Import
        import models
        
        # Create args (like throughput.py and main.py do)
        class Args:
            config = 'carv4'
            sa = model_type in ['small', 'full']  # tiny: False, others: True
            dil = model_type in ['small', 'full']
        
        args = Args()
        print(f"  config={args.config}, sa={args.sa}, dil={args.dil}")
        
        # Map to actual model names in models/__dict__
        # Based on scripts.sh:
        # - pidinet (with --sa --dil for full)
        # - pidinet (without --sa --dil for tiny)  
        # - pidinet_small (with --sa --dil)
        
        if model_type == 'tiny':
            model_name = 'pidinet'  # No SA/DIL
        elif model_type == 'small':
            model_name = 'pidinet_small'  # With SA/DIL
        else:  # full
            model_name = 'pidinet'  # With SA/DIL
        
        # Create model
        model_fn = models.__dict__[model_name]
        model = model_fn(args)
        
        # Load checkpoint
        if checkpoint_path:
            print(f"  Loading {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            state = remove_module_prefix(state)
            model.load_state_dict(state)
            print("  ✓ Loaded")
        
        # NO CONVERSION - use original PDC model for benchmarking!
        # (conversion is only for deployment/ONNX export)
        
        self.model = model
        
        # Precision
        if precision == 'fp16':
            self.model = self.model.half()
        
        self.model.to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model ready: {self.num_params:,} params\n")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def benchmark_folder(self, folder, warmup=20, iters=100, save_log=True):
        """Benchmark on folder"""
        
        # Load images
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            files.extend(glob.glob(f"{folder}/{ext}"))
        
        if not files:
            raise FileNotFoundError(f"No images in {folder}")
        
        print(f"Loading {len(files)} images...")
        tensors, sizes = [], []
        
        for f in tqdm(files):
            try:
                img = Image.open(f).convert('RGB')
                sizes.append(img.size)
                t = self.transform(img).unsqueeze(0)
                if self.precision == 'fp16':
                    t = t.half()
                tensors.append(t)
            except:
                pass
        
        data = cycle(tensors)
        
        # Warmup
        print(f"Warmup ({warmup})...")
        with torch.no_grad():
            for _ in range(warmup):
                self.model(next(data).to(self.device))
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking ({iters})...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        
        with torch.no_grad():
            for _ in tqdm(range(iters)):
                inp = next(data).to(self.device)
                s.record()
                self.model(inp)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        
        # Stats
        lat = np.mean(times)
        std = np.std(times)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        fps = 1000 / lat
        vram = torch.cuda.max_memory_allocated() / 1024**2
        w, h = int(np.mean([s[0] for s in sizes])), int(np.mean([s[1] for s in sizes]))
        
        # Report
        report = f"""
{'='*70}
{self.model_type.upper()} - {self.precision.upper()}
{'='*70}
Images:     {len(files)} ({w}×{h})
Latency:    {lat:.2f} ± {std:.2f} ms
P95/P99:    {p95:.2f} / {p99:.2f} ms
FPS:        {fps:.2f}
VRAM:       {vram:.2f} MB
Params:     {self.num_params:,}
{'='*70}
"""
        print(report)
        
        if save_log:
            with open(f"bench_{self.model_type}_{self.precision}.txt", 'w') as f:
                f.write(report)
            print("✓ Log saved\n")
        
        return {
            'latency_ms': lat,
            'std_ms': std,
            'p95_ms': p95,
            'p99_ms': p99,
            'fps': fps,
            'vram_mb': vram
        }
    
    def stress_test(self, image_path, resolutions=None, save_log=True):
        """Stress test"""
        if resolutions is None:
            resolutions = [1024, 2048, 4096, 8192]
        
        img = Image.open(image_path).convert('RGB')
        print(f"\nStress: {self.model_type.upper()}\n{'-'*40}")
        results = []
        
        for size in resolutions:
            try:
                print(f"{size}×{size}...", end=' ')
                t = self.transform(img.resize((size, size))).unsqueeze(0).to(self.device)
                if self.precision == 'fp16':
                    t = t.half()
                
                with torch.no_grad():
                    for _ in range(5):
                        self.model(t)
                torch.cuda.synchronize()
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                times = []
                
                with torch.no_grad():
                    for _ in range(20):
                        s.record()
                        self.model(t)
                        e.record()
                        torch.cuda.synchronize()
                        times.append(s.elapsed_time(e))
                
                lat = np.mean(times)
                vram = torch.cuda.max_memory_allocated() / 1024**2
                print(f"✓ {lat:.1f}ms, {vram:.0f}MB")
                results.append({'resolution': size, 'latency_ms': lat, 'vram_mb': vram, 'status': 'OK'})
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("✗ OOM")
                    results.append({'resolution': size, 'status': 'OOM'})
                    torch.cuda.empty_cache()
                    break
                raise
        
        print()
        return results


def bench(model='tiny', folder='test_images/', ckpt=None):
    """Quick bench"""
    b = PiDiNetBenchmarker(model, 'fp16', ckpt)
    return b.benchmark_folder(folder)


def stress(model='tiny', image='test.jpg', ckpt=None):
    """Quick stress"""
    b = PiDiNetBenchmarker(model, 'fp16', ckpt)
    return b.stress_test(image)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='tiny', choices=['tiny', 'small', 'full'])
    p.add_argument('--precision', default='fp16', choices=['fp32', 'fp16'])
    p.add_argument('--mode', default='folder', choices=['folder', 'stress'])
    p.add_argument('--input', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=100)
    a = p.parse_args()
    
    b = PiDiNetBenchmarker(a.model, a.precision, a.checkpoint)
    if a.mode == 'folder':
        b.benchmark_folder(a.input, a.warmup, a.iters)
    else:
        b.stress_test(a.input)
