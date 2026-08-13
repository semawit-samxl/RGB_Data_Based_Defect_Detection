import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "models/training_metrics.csv"
)

plt.figure(figsize=(12, 6))

plt.plot(
    df["epoch"],
    df["total_loss"],
    label="Total Loss",
    linewidth=3
)

plt.plot(
    df["epoch"],
    df["loss_classifier"],
    label="Classifier"
)

plt.plot(
    df["epoch"],
    df["loss_box_reg"],
    label="Box Regression"
)

plt.plot(
    df["epoch"],
    df["loss_objectness"],
    label="Objectness"
)

plt.plot(
    df["epoch"],
    df["loss_rpn_box_reg"],
    label="RPN Box Reg"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Faster R-CNN Training Losses")
plt.legend()
plt.grid(True)

plt.show()