import yaml 
from ultralytics import YOLO
import os

def train_model():

   with open ("configs/params.yaml","r") as file:
      params= yaml.safe_load(file)

      print("Import Config File")

      model=YOLO(params["training"]["model"])

      model.train(
        data="configs/data.yaml",
        epochs=params["training"]["epochs"],
        imgsz=params["training"]["image_size"],
        batch=params["training"]["batch_size"],
        patience=params["training"]["patience"],
        device=params["training"]["device"]
    )
      print("Complete Model Training")
      
if __name__ == "__main__":
   train_model()
