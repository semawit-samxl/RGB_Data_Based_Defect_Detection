from ultralytics import YOLO

def evaluate():

    model = YOLO("runs/detect/train/weights/best.pt")

    metrics = model.val(
        data="configs/data.yaml"
    )

    return metrics


if __name__ == "__main__":
    metrics = evaluate()
    print(metrics)