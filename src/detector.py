from ultralytics import YOLO
import cv2
import os

class VehicleDetector:
    def __init__(self, model_name='yolo11n.pt'):
        # Load pre-trained YOLO11 model
        self.model = YOLO(model_name)
        # Class names for COCO (0=person, 2=car, 3=motorcycle, 5=bus, 7=truck)
        self.vehicle_classes = [2, 3, 5, 7]
        print(f"Vehicle Detector ({model_name}) initialized.")

    def detect_vehicles(self, image):
        """Detects vehicles in a frame."""
        results = self.model(image, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    # Get box coordinates (x1, y1, x2, y2)
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    label = self.model.names[cls_id]
                    
                    detections.append({
                        "bbox": coords,
                        "confidence": conf,
                        "class": label,
                        "class_id": cls_id
                    })
        
        return detections

if __name__ == "__main__":
    detector = VehicleDetector()
    # Test on the mock image generated earlier
    sample_img_path = "data/mock_plate.jpg"
    if os.path.exists(sample_img_path):
        img = cv2.imread(sample_img_path)
        results = detector.detect_vehicles(img)
        
        print(f"\nDetection results for {sample_img_path}:")
        for d in results:
            print(f"Found {d['class']} with confidence {d['confidence']:.2f} at {d['bbox']}")
    else:
        print(f"Sample image {sample_img_path} not found.")
