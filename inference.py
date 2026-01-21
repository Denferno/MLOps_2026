import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from ml_core.models import MLP


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    model = MLP(
        input_shape=config["data"]["input_shape"],
        hidden_units=config["model"]["hidden_units"],
        dropout_rate=config["model"]["dropout_rate"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def preprocess_image(image_path: Path):
    tfm = T.Compose(
        [
            T.Resize((96, 96)),
            T.ToTensor(),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0)  # (1,3,96,96)
    return x


def main():
    parser = argparse.ArgumentParser(description="Run inference with a saved MLP checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt/.pth")
    parser.add_argument("--image", type=Path, required=True, help="Path to an image (png/jpg)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_from_checkpoint(args.checkpoint, device)
    x = preprocess_image(args.image).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    print(f"Predicted class: {pred}")
    print(f"Probabilities: {probs.detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
