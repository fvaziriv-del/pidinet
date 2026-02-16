"""
Universal PiDiNet Benchmarker - Works with ANY signature!
Just paste this in your notebook
"""

import torch
import glob
import numpy as np
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
        
        print(f"Loading {model_type} model...")
        
        # Import
        import models
        from models.convert_pidinet import convert_pidinet
        
        # Map model_type to actual model name in the repo
        model_map = {
            'tiny': 'pidinet',        # tiny is just pidinet without --sa --dil
            'small': 'pidinet_small', # small uses pidinet_small
            'full': 'pidinet',        # full is pidinet with --sa --dil
        }
        
        model_name = model_map[model_type]
        
        # Create model using the same way main.py does it
        # main.py line ~230: model = models.__dict__[args.model](args)
        # So the model takes args object!
        
        # Create a fake args object
        class Args:
            pass
        
        args = Args()
        args.config = 'carv4'
        
        # Set sa and dil based on model_type
        if model_type == 'tiny':
            args.sa = False
            args.dil = False
        else:  # small or full
            args.sa = True
            args.dil = True
        
        # Get model function
        model_fn = models.__dict__[model_name]
        
        # Create model - it takes args object
        print(f"  Model: {model_name}, config: {args.config}, sa: {args.sa}, dil: {args.dil}")
        raw_model = model_fn(args)
        
        # Load checkpoint
        if checkpoint_path:
            print(f"  Loading checkpoint...")
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            raw_model.load_state_dict(state)
        
        # Convert
        print("  Converting...")
        self.model = convert_pidinet(raw_model, args.config)
        
        # Precision
        if precision == 'fp16':
            self.model = self.model.half()
        
        self.model.to(self.device).eval()
        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Ready! {self.num_params:,} params\n")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def benchmark_folder(self, folder, warmup=20, iters=100):
        """Benchmark on images"""
        
        # Load
        files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            files.extend(glob.glob(f"{folder}/{ext}"))
        
        print(f"Loading {len(files)} images...")
        tensors, sizes = [], []
        
        for f in tqdm(files):
            img = Image.open(f).convert('RGB')
            sizes.append(img.size)
            t = self.transform(img).unsqueeze(0)
            if self.precision == 'fp16':
                t = t.half()
            tensors.append(t)
        
        data = cycle(tensors)
        
        # Warmup
        print(f"Warmup...")
        with torch.no_grad():
            for _ in range(warmup):
                self.model(next(data).to(self.device))
        torch.cuda.synchronize()
        
        # Benchmark
        print(f"Benchmarking...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        times = []
        
        with torch.no_grad():
            for _ in tqdm(range(iters)):
                inp = next(data).to(self.device)
                s.record()
                self.model(inp)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
        
        # Results
        lat = np.mean(times)
        std = np.std(times)
        p95 = np.percentile(times, 95)
        fps = 1000 / lat
        vram = torch.cuda.max_memory_allocated() / 1024**2
        
        w = int(np.mean([s[0] for s in sizes]))
        h = int(np.mean([s[1] for s in sizes]))
        
        print(f"\n{'='*60}")
        print(f"{self.model_type.upper()} - {self.precision.upper()}")
        print(f"{'='*60}")
        print(f"Images:    {len(files)} ({w}×{h})")
        print(f"Latency:   {lat:.2f} ± {std:.2f} ms (p95: {p95:.2f})")
        print(f"FPS:       {fps:.2f}")
        print(f"VRAM:      {vram:.2f} MB")
        print(f"Params:    {self.num_params:,}")
        print(f"{'='*60}\n")
        
        return {'latency': lat, 'fps': fps, 'vram': vram}
    
    def stress_test(self, image, resolutions=None):
        """Stress test"""
        
        if resolutions is None:
            resolutions = [1024, 2048, 4096]
        
        img = Image.open(image).convert('RGB')
        print(f"\nStress Test: {self.model_type.upper()}\n{'-'*40}")
        
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
                
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
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
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print("✗ OOM")
                    torch.cuda.empty_cache()
                    break
                raise
        print()


# Quick functions
def bench(model='tiny', folder='test_images/', ckpt=None):
    b = PiDiNetBenchmarker(model, 'fp16', ckpt)
    return b.benchmark_folder(folder)

def stress(model='tiny', image='test.jpg', ckpt=None):
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
    args = p.parse_args()
    
    b = PiDiNetBenchmarker(args.model, args.precision, args.checkpoint)
    
    if args.mode == 'folder':
        b.benchmark_folder(args.input, args.warmup, args.iters)
    else:
        b.stress_test(args.input)
