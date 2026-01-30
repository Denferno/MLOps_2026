import time
import torch
from ml_core.models.mlp import MLP

def measure_throughput(model, device, batch_size=1, max_batches=200):
    model.eval()
    warmup_batches = min(20, max_batches)
    x = torch.rand(batch_size, 3, 96, 96, device=device)

    with torch.no_grad():
        for _ in range(warmup_batches):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    images = 0

    with torch.no_grad():
        for b in range(max_batches):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    images = batch_size * max_batches
    img_ps = images / elapsed
    return img_ps, elapsed


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_shape=(3, 96, 96), hidden_units=[128]).to(device)

    img_ps, elapsed = measure_throughput(model, device, batch_size=1, max_batches=200)
    print(f"device={device.type} img_ps={img_ps:.2f} elapsed_s={elapsed:.2f}")