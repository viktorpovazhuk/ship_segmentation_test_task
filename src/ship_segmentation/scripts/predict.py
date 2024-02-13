from pathlib import Path

import matplotlib.pyplot as plt

import cv2

import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import argparse

import time

from ship_segmentation.models.unet import UNetLitModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    # TODO: implement thresholding
    parser.add_argument("--cls_threshold", type=float, default=0.3)

    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--device_number", type=int, default=1)

    parser.add_argument("--output_path", default="output/predictions/")

    parser.add_argument("--ckpt_path", default="output/ckpts/last.ckpt")

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_number}")
        else:
            print("GPU is not available. Default to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    model = UNetLitModel.load_from_checkpoint(args.ckpt_path).to(device)

    img = cv2.imread(args.image_path)

    transform = A.Compose(
        [
            A.Resize(height=384, width=384),  # 224, 576
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    transformed_img = transform(image=img)["image"]
    transformed_img = transformed_img.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(transformed_img)

    pred = torch.max(output, 1)[1].cpu().numpy()

    fig = plt.figure(figsize=(6, 3))

    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    ax.imshow(img)
    ax.set_title(f"img", fontsize=10)
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    ax.imshow(pred[0])
    ax.set_title(f"pred", fontsize=10)

    plt.savefig(output_path / f"pred_{time.time()}.png")


if __name__ == "__main__":
    print("Predicting...")
    main()
