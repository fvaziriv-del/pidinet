%%writefile benchmark_pidinet.py
import sys
import os
# Fix for ImportError: ensuring current directory is prioritized for 'utils'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

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

# Internal repository imports
try:
    from utils import convert_pidinet
    from models.pidinet import pidinet
except ImportError:
    # Fallback for direct import if sys.path isn't enough
    import utils
    from models.pidinet import pidinet
    convert_pidinet = utils.convert_pidinet

class PiDiNetBenchmarker:
    def __init__(self, model_type='tiny', precision='fp32', use_converted=True, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("Benchmark requires an NVIDIA GPU.")
        
        self.model_type = model_type
        self.precision = precision
        self.use_converted = use_converted
        
        config = 'ant' if model_type == 'tiny' else 'carv4'
        sa = True if model_type == 'full' else False
        dil = True if model_type == 'full' else False
        
        raw_model = pidinet(config=config, sa=sa, dil=dil)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            raw_model.load_state_dict(state_dict)
        
        if use_converted:
            self.model = convert_pidinet(raw_model, config)
        else:
            self.model = raw_model
            
        if self.precision == 'fp16':
            self.model = self.model.half()
            
        self.model.to(self.device).eval()
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def run_folder_benchmark(self, folder_path, warmup_iters=20, test_iters=100, save_log=True):
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for e in exts: files.extend(glob.glob(os.path.join(folder_path, e)))
        if not files: raise FileNotFoundError(f"No images found in {folder_path}")
            
        tensors = []
        res_list = []
        for f in files:
            img = Image.open(f).convert('RGB')
            res_list.append(img.size)
            t = self.transform(img).unsqueeze(0)
            if self.precision == 'fp16': t = t.half()
            tensors.append(t)
            
        data_pool = cycle(tensors)
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = self.model(next(data_pool).to(self.device))
        
        torch.cuda.synchronize()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_init = torch.cuda.memory_allocated() / 1024**2

        with torch.no_grad():
            for _ in range(test_iters):
                inp = next(data_pool).to(self.device)
                starter.record()
                _ = self.model(inp)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        avg_lat = np.mean(timings)
        fps = 1000 / avg_lat
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        avg_w, avg_h = int(np.mean([r[0] for r in res_list])), int(np.mean([r[1] for r in res_list]))
        
        report = (
            f"{'='*60}\nFOLDER BENCHMARK: {self.model_type} | {self.precision}\n{'='*60}\n"
            f"Date: {datetime.now()}\nRes: {avg_w}x{avg_h} | Latency: {avg_lat:.3f} ms | FPS: {fps:.2f}\n"
            f"VRAM: {peak_vram:.2f} MB | Params: {self.total_params:,}\n{'='*60}\n"
        )
        print(report)
        if save_log:
            log_name = f"bench_folder_{self.model_type}_{self.precision}.txt"
            with open(log_name, 'w') as f: f.write(report)
            print(f"Log saved: {log_name}")

    def run_stress_test(self, image_path, save_log=True):
        resolutions = [1024, 2048, 4096, 8192]
        img = Image.open(image_path).convert('RGB')
        print(f"\nSTRESS TEST: {self.model_type.upper()} | {self.precision.upper()}\n{'-'*60}")
        
        log_entries = []
        for side in resolutions:
            try:
                img_resized = img.resize((side, side))
                t = self.transform(img_resized).unsqueeze(0).to(self.device)
                if self.precision == 'fp16': t = t.half()
                with torch.no_grad():
                    for _ in range(5): _ = self.model(t)
                torch.cuda.synchronize()
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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
                status = f"RES: {side}x{side} | Latency: {np.mean(latencies):.2f} ms | VRAM: {vram:.2f} MB | PASSED"
                print(status)
                log_entries.append(status)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"RES: {side}x{side} | FAILED (OOM)")
                    break
        
        if save_log:
            log_name = f"stress_{self.model_type}_{self.precision}.txt"
            with open(log_name, 'w') as f: f.write("\n".join(log_entries))
            print(f"Log saved: {log_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tiny')
    parser.add_argument('--precision', type=str, default='fp16')
    parser.add_argument('--mode', type=str, default='stress')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--save', type=str, default='True', choices=['True', 'False'])

    args = parser.parse_args()
    save_bool = args.save == 'True'

    bench = PiDiNetBenchmarker(args.model, args.precision, True, args.checkpoint)
    
    if args.mode == 'stress':
        bench.run_stress_test(args.input, save_log=save_bool)
    else:
        bench.run_folder_benchmark(args.input, args.warmup, args.iters, save_log=save_bool)
