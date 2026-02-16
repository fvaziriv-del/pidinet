import torch
import torch.nn as nn
import time
import os
import glob
import platform
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from itertools import cycle

# Internal repository imports
from models.pidinet import pidinet
from utils import convert_pidinet

class PiDiNetBenchmarker:
    def __init__(self, model_type='tiny', precision='fp32', use_converted=True, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("Benchmark requires an NVIDIA GPU and CUDA.")
        
        self.model_type = model_type
        self.precision = precision
        self.use_converted = use_converted
        self.dataset_name = "BSDS500"
        
        # Architecture mapping
        config = 'ant' if model_type == 'tiny' else 'carv4'
        sa = True if model_type == 'full' else False
        dil = True if model_type == 'full' else False
        
        # Initialize model
        raw_model = pidinet(config=config, sa=sa, dil=dil)
        
        # Load weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            raw_model.load_state_dict(state_dict)
        
        # Convert to optimized Vanilla Convolution
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

    def _get_hw_info(self):
        return {
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "os": f"{platform.system()} {platform.release()}"
        }

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
        
        hw = self._get_hw_info()
        report = (
            f"{'='*60}\n              FOLDER INFERENCE BENCHMARK\n{'='*60}\n"
            f"[SYSTEM]  Date: {datetime.now()} | GPU: {hw['gpu']}\n"
            f"[MODEL]   Arch: PiDiNet ({self.model_type}) | Precision: {self.precision.upper()}\n"
            f"[METRICS] Avg Res: {avg_w}x{avg_h} | Latency: {avg_lat:.3f} ms | FPS: {fps:.2f} | VRAM: {peak_vram:.2f} MB\n"
            f"{'='*60}\n"
        )
        print(report)
        if save_log:
            with open(f"bench_folder_{self.model_type}_{self.precision}.txt", 'w') as f: f.write(report)

    def run_stress_test(self, image_path, save_log=True):
        resolutions = [1024, 2048, 4096, 8192]
        img = Image.open(image_path).convert('RGB')
        hw = self._get_hw_info()
        print(f"\nSTRESS TEST: {self.model_type.upper()} | Precision: {self.precision.upper()} | GPU: {hw['gpu']}")
        
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
                
                avg_ms = np.mean(latencies)
                vram = torch.cuda.max_memory_allocated() / 1024**2
                status = f"RES: {side}x{side} | Latency: {avg_ms:.2f} ms | VRAM: {vram:.2f} MB | STATUS: PASSED"
                print(status)
                log_entries.append(status)
                del t
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    status = f"RES: {side}x{side} | STATUS: FAILED (CUDA Out of Memory)"
                    print(status)
                    log_entries.append(status)
                    torch.cuda.empty_cache()
                    break
                else: raise e

        if save_log:
            with open(f"stress_{self.model_type}_{self.precision}.txt", 'w') as f:
                f.write("\n".join(log_entries))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PiDiNet Performance Benchmarking Tool")
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'full'], help="Model variant")
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp16', 'fp32'], help="Floating point precision")
    parser.add_argument('--mode', type=str, default='stress', choices=['stress', 'folder'], help="Benchmark mode")
    parser.add_argument('--input', type=str, required=True, help="Path to image (stress) or folder (folder)")
    parser.add_argument('--checkpoint', type=str, help="Path to .pth checkpoint")
    parser.add_argument('--warmup', type=int, default=20, help="Number of warmup iterations")
    parser.add_argument('--iters', type=int, default=100, help="Number of test iterations")

    args = parser.parse_args()

    bench = PiDiNetBenchmarker(
        model_type=args.model, 
        precision=args.precision, 
        use_converted=True,
        checkpoint_path=args.checkpoint
    )
    
    if args.mode == 'stress':
        bench.run_stress_test(args.input)
    else:
        bench.run_folder_benchmark(args.input, warmup_iters=args.warmup, test_iters=args.iters)
