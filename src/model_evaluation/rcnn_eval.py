import yaml
import torch
import os
import json

from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.data_preparation.faster_rcnn_dataset import (
    ThermalDataset,
    collate_fn
)

from src.model_training.faster_rcnn_model import (
    create_model
)


def evaluate():

    # Load configuration
    with open(
        "configs/rcnn_params.yaml",
        "r"
    ) as file:

        params = yaml.safe_load(file)

    # Select device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Create validation dataset
    dataset = ThermalDataset(
        image_dir=
        f"{params['data']['processed_data_path']}/images/val",

        label_dir=
        f"{params['data']['processed_data_path']}/labels/val"
    )

    # Create validation dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=params["evaluation"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    model = create_model()

    # Load checkpoint
    checkpoint = torch.load(
        "models/best_faster_rcnn.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    # Evaluation metric
    metric = MeanAveragePrecision(
    class_metrics=True
         )

    with torch.no_grad():

        for images, targets in dataloader:

            images = [
                image.to(device)
                for image in images
            ]

            predictions = model(images)

            predictions = [
                {
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred["labels"].cpu()
                }
                for pred in predictions
            ]

            targets = [
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                }
                for target in targets
            ]

            metric.update(
                predictions,
                targets
            )

    results = metric.compute()

    CLASS_NAMES = {
        1: "Blob",
        2: "Underextrusion",
        3: "Sagging"
    }

    classes = results["classes"].tolist()
    map_per_class = results["map_per_class"].tolist()
    mar_per_class = results["mar_100_per_class"].tolist()

    print("\nClass-wise Performance")
    print("-" * 50)

    for cls, ap, recall in zip(
        classes,
        map_per_class,
        mar_per_class
    ):
        print(
            f"{CLASS_NAMES[cls]:15s} "
            f"AP={ap:.4f} "
            f"={recall:.4f}"
        )

    # Create evaluation directory
    os.makedirs(
        "artifacts/evaluation",
        exist_ok=True
    )

    # Create JSON dictionary
    results_dict = {}

    for key, value in results.items():

        if torch.is_tensor(value):

            if value.numel() == 1:
                results_dict[key] = float(
                    value.item()
                )
            else:
                results_dict[key] = (
                    value.tolist()
                )

        else:
            results_dict[key] = value

    results_dict["class_metrics"] = []

    for cls, ap, recall in zip(
        classes,
        map_per_class,
        mar_per_class
    ):
        results_dict["class_metrics"].append(
            {
                "class_name": CLASS_NAMES[cls],
                "ap": float(ap),
                "recall": float(recall)
            }
        )

    # Save metrics
    with open(
        "artifacts/evaluation/faster_rcnn_metrics.json",
        "w"
    ) as file:

        json.dump(
            results_dict,
            file,
            indent=4
        )

    print("\nEvaluation Results")
    print("-" * 30)

    print(f"mAP: {float(results['map']):.4f}")

    print(f"mAP50: {float(results['map_50']):.4f}")

    print(f"mAP75: {float(results['map_75']):.4f}" )

    print(f"Mean Recall: {float(results['mar_100']):.4f}" )

    return results


def main():

    evaluate()


if __name__ == "__main__":

    main()