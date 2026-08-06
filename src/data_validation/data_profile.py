import os
from collections import defaultdict

base_path="data/interim"
datasets=["rgb_yolo_priint3",
          "rgb_yolo_print_1",
          "rgb_yolo_print_2"
          ]

for dataset in datasets:
    data_path=os.path.join(base_path,dataset,"obj_train_data")
    images=[f for f in os.listdir(data_path) if f.endswith(".png")]
    labels=[f  for f in os.listdir(data_path) if f.endswith (".txt")]

    # get annotations
    annotated=0
    for label in labels:
        label_path=os.path.join(data_path,label)
        if os.path.getsize(label_path)>0:
            annotated+=1
    print(f"/n{dataset}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Annoatetd Frames: {annotated}")

