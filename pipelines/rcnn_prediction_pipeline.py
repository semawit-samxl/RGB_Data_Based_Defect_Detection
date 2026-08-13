import os 
import sys

sys.path.append(os.getcwd())

from src.model_prediction.predict_rcnn import predict

if __name__ == "__main__":

    predict()

print("Model Prediction Done !")