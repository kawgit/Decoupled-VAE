"""A run's logs: its summary, and a .full.log twin that also tracks the GPU's use."""

import os
import threading
import time

import torch


class RunLog:
    """Called with a message, prints it and appends it, timestamped, to path and to path's .full.log twin. On a
    CUDA device, the twin also gets a line every interval seconds: the GPU's utilization averaged over the
    interval (sampled each second; needs nvidia-ml-py) and its memory, the GPU's as a whole and this run's."""

    def __init__(self, path, device, interval=30.0):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.summary = open(path, "a")
        self.full = open(os.path.splitext(path)[0] + ".full.log", "a")
        self.lock = threading.Lock()
        if device.type == "cuda" and interval > 0:
            threading.Thread(target=self.watch, args=(device, interval), daemon=True).start()

    def write(self, msg, summary=True):
        msg = f"{time.strftime('%H:%M:%S')} {msg}"
        with self.lock:
            for f in (self.summary, self.full) if summary else (self.full,):
                f.write(msg + "\n")
                f.flush()
        return msg

    def __call__(self, msg):
        print(self.write(msg), flush=True)

    def watch(self, device, interval):
        gib = 2 ** 30
        while True:
            samples, end = [], time.time() + interval
            while time.time() < end:
                time.sleep(1.0)
                try:
                    samples.append(torch.cuda.utilization(device))
                except Exception:  # no NVML: memory only
                    pass
            free, total = torch.cuda.mem_get_info(device)
            util = f"{sum(samples) / len(samples):.0f}% util (min {min(samples)}%, max {max(samples)}%), " \
                if samples else ""
            self.write(f"gpu: {util}{(total - free) / gib:.1f} of {total / gib:.1f} GB used; this run "
                       f"{torch.cuda.memory_allocated(device) / gib:.1f} GB allocated (peak "
                       f"{torch.cuda.max_memory_allocated(device) / gib:.1f}), "
                       f"{torch.cuda.memory_reserved(device) / gib:.1f} GB reserved", summary=False)
