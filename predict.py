r"""Local inference for the digit bundle exported in notebooks 13 and 14.

Example: .venv\Scripts\python.exe predict.py artifacts\13_demo_digit.png
This model expects a centered LIGHT digit on a DARK background.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from course_utils import TinyCNN, ARTIFACTS


def load_bundle(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("architecture") != "TinyCNN-v1":
        raise ValueError("Expected a TinyCNN-v1 course bundle.")
    if bundle.get("channels") != 1 or bundle.get("image_size") != 8:
        raise ValueError("This predictor supports 8x8 grayscale digit bundles only.")
    model = TinyCNN(num_classes=len(bundle["classes"]))
    model.load_state_dict(bundle["state_dict"])
    return model.to(device).eval(), bundle


def image_to_tensor(path, image_size=8, invert=False):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        if invert:
            image = ImageOps.invert(image)
        # Match the digits preprocessing recorded in the exported bundle.
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def predict_image(model, bundle, path, top_k=3, invert=False):
    device = next(model.parameters()).device
    x = image_to_tensor(path, bundle["image_size"], invert).to(device)
    probabilities = model(x).softmax(1)[0].cpu()
    values, indices = probabilities.topk(min(top_k, len(probabilities)))
    return [{"class": bundle["classes"][i], "probability": float(p)}
            for p, i in zip(values, indices.tolist())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=ARTIFACTS / "13_digit_classifier.pt")
    parser.add_argument("--invert", action="store_true",
                        help="Use for a dark digit on a light background")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not args.checkpoint.exists():
        parser.error("Run notebook 13 first to create the checkpoint.")
    model, bundle = load_bundle(args.checkpoint)
    print(json.dumps(predict_image(model, bundle, args.image,
                                   args.top_k, args.invert), indent=2))


if __name__ == "__main__":
    main()
