import time
import torch

from time import perf_counter
from ml_core.data.loader import get_dataloaders
from ml_core.models.model import MLP

def measure_throughput(dataloader, model, device, max_batches=20):
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, _ = get_dataloaders(batch_size=1, num_workers=4)
    model = MLP()
    model.to(device)

    img_ps, elapsed = measure_throughput(
        train_loader,
        model,
        device,
        max_batches=200
    )

    print(f"device={device.type} img_s={img_s:.2f} elapsed_s={elapsed:.2f}")
