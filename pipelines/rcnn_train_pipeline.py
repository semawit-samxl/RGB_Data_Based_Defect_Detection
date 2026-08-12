import sys
import os 
sys.path.append(os.getcwd())

from src.model_training.rcnn_train import ( train_model)

if __name__ == "__main__":
    train_model()