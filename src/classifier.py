from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import cv2
import numpy as np

class VehicleMakeModelClassifier:
    def __init__(self, model_name="dima806/car_models_image_detection"):
        """
        Initialize the vehicle make/model classifier.
        Using a lightweight model from HuggingFace.
        """
        print(f"Loading vehicle classifier: {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.eval()
        print("Vehicle classifier loaded.")

    def classify(self, image):
        """
        Classify a vehicle image to determine make/model.
        Args:
            image: numpy array (BGR format from cv2) or PIL Image
        Returns:
            dict with 'make_model' and 'confidence'
        """
        # Convert BGR to RGB if numpy array
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Process and predict
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probabilities[0][predicted_class_idx].item()
        
        # Get the label
        label = self.model.config.id2label[predicted_class_idx]
        
        return {
            "make_model": label,
            "confidence": confidence
        }

if __name__ == "__main__":
    # Test initialization
    classifier = VehicleMakeModelClassifier()
    print("Classifier ready for inference.")
