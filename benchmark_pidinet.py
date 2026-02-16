import torch
import torch.nn as nn
import time
import os
import glob
import platform
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
        self.dataset_name = "BSDS500" # Fixed as requested
        
        # Architecture configuration
        config = 'ant' if model_type == 'tiny' else 'carv4'
        sa = True if model_type == 'full' else False
        dil = True if model_type == 'full' else False
        
        # Model Initialization
        raw_model = pidinet(config=config, sa=sa, dil=dil)
        
        # Load Weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            raw_model.load_state_dict(state_dict)
        
        # Optimization: Convert to Vanilla Convolution
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

    def _prepare_data(self, folder_path):
        """Loads all images from folder into RAM to isolate I/O lag."""
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder_path, e)))
        
        if not files:
            raise FileNotFoundError(f"No images found in {folder_path}")
            
        tensors = []
        resolutions = []
        for f in files:
            img = Image.open(f).convert('RGB')
            resolutions.append(img.size)
            t = self.transform(img).unsqueeze(0)
            if self.precision == 'fp16':
                t = t.half()
            tensors.append(t)
            
        return tensors, resolutions

    def run_benchmark(self, folder_path, warmup_iters=20, test_iters=100, save_log=True):
        tensors, res_list = self._prepare_data(folder_path)
        data_pool = cycle(tensors)
        
        # Hardware Info
        gpu_name = torch.cuda.get_device_name(0)
        
        # Warm-up Phase
        print(f"Starting Warm-up ({warmup_iters} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iters):
                input_tensor = next(data_pool).to(self.device)
                _ = self.model(input_tensor)
        
        torch.cuda.synchronize()
        
        # Timing Setup
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []
        
        # Memory Measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_initial = torch.cuda.memory_allocated() / 1024**2

        # Pure Inference Loop
        print(f"Running Main Benchmark ({test_iters} iterations)...")
        with torch.no_grad():
            for _ in range(test_iters):
                input_tensor = next(data_pool).to(self.device)
                
                starter.record()
                _ = self.model(input_tensor)
                ender.record()
                
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        # Statistics
        avg_lat = np.mean(timings)
        std_lat = np.std(timings)
        fps = 1000 / avg_lat
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)
        peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        
        # Average Resolution for Log
        avg_w = int(np.mean([r[0] for r in res_list]))
        avg_h = int(np.mean([r[1] for r in res_list]))
        mp = (avg_w * avg_h) / 1e6

        # Report Construction
        report = (
            f"{'='*60}\n"
            f"              PIDINET PERFORMANCE BENCHMARK\n"
            f"{'='*60}\n"
            f"[SYSTEM INFORMATION]\n"
            f"Date:                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Hardware:            {gpu_name}\n"
            f"CUDA Version:        {torch.version.cuda}\n"
            f"PyTorch Version:     {torch.__version__}\n"
            f"Platform:            {platform.system()} {platform.release()}\n\n"
            f"[MODEL SPECIFICATIONS]\n"
            f"Architecture:        PiDiNet\n"
            f"Config Profile:      {'ant' if self.model_type=='tiny' else 'carv4'}\n"
            f"Pretrained Dataset:  {self.dataset_name}\n"
            f"Converted Model:     {'Yes' if self.use_converted else 'No'}\n"
            f"Precision:           {self.precision.upper()}\n"
            f"Total Parameters:    {self.total_params:,}\n\n"
            f"[BENCHMARK CONFIGURATION]\n"
            f"Input Folder:        {folder_path} ({len(res_list)} unique images)\n"
            f"Avg Resolution:      {avg_w} x {avg_h} (~{mp:.2f} MP)\n"
            f"Warm-up Iterations:  {warmup_iters}\n"
            f"Test Iterations:     {test_iters}\n\n"
            f"[PERFORMANCE METRICS]\n"
            f"--- Latency Statistics ---\n"
            f"Average Latency:     {avg_lat:.3f} ms\n"
            f"Median Latency:      {np.median(timings):.3f} ms\n"
            f"Min / Max Latency:   {np.min(timings):.3f} ms / {np.max(timings):.3f} ms\n"
            f"Std Deviation:       {std_lat:.4f} ms\n"
            f"P95 / P99 Latency:   {p95:.3f} ms / {p99:.3f} ms\n\n"
            f"--- Throughput ---\n"
            f"Inference FPS:       {fps:.2f} frames/sec\n"
            f"Pixel Throughput:    {fps * mp:.2f} Megapixels/sec\n\n"
            f"--- Memory (VRAM) ---\n"
            f"Initial Usage:       {mem_initial:.2f} MB\n"
            f"Peak Allocated:      {peak_vram:.2f} MB\n"
            f"Inference Delta:     {peak_vram - mem_initial:.2f} MB\n"
            f"{'='*60}\n"
        )
        
        print(report)
        
        if save_log:
            log_path = os.path.join(folder_path, f"benchmark_log_{self.precision}.txt")
            with open(log_path, 'w') as f:
                f.write(report)
            print(f"Log saved to: {log_path}")

if __name__ == "__main__":
    # Settings
    bench = PiDiNetBenchmarker(
        model_type='tiny',         # Options: 'tiny', 'full'
        precision='fp16',          # Options: 'fp16', 'fp32'
        use_converted=True,        # Use optimized vanilla conv
        checkpoint_path='./checkpoints/table5_pidinet-tiny.pth'
    )
    
    # Execution
    bench.run_benchmark(
        folder_path='./benchmark_images', 
        warmup_iters=30, 
        test_iters=200, 
        save_log=True
    )