import time
import torch

def measure_throughput(dataloader, model, device, max_batches=200):
    model.eval()
    warmup_batches = min(20, max_batches)

    it = iter(dataloader)
    with torch.no_grad():
        for _ in range(warmup_batches):
            x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    images = 0

    it = iter(dataloader)
    with torch.no_grad():
        for b in range(max_batches):
            x, y = next(it)
            images += x.shape[0]

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    img_ps = images / elapsed
    return img_ps, elapsed