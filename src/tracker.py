import cv2
from ultralytics import YOLO
import os

class VehicleTracker:
    def __init__(self, model_name='yolo11n.pt'):
        # Load YOLO model
        self.model = YOLO(model_name)
        # Unique vehicle IDs tracked
        self.tracked_ids = set()
        print(f"Vehicle Tracker ({model_name}) initialized with ByteTrack.")

    def track_and_count(self, frame):
        """Processes a frame, tracks vehicles, and updates the count."""
        # Use persist=True to maintain tracks across frames
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                # Class filter: Car, Motorcycle, Bus, Truck
                if int(cls) in [2, 3, 5, 7]:
                    self.tracked_ids.add(track_id)
                    detections.append({
                        "id": track_id,
                        "bbox": box.tolist(),
                        "class": self.model.names[int(cls)],
                        "confidence": float(conf)
                    })
        
        return detections, len(self.tracked_ids)

if __name__ == "__main__":
    # Test on a dummy video or image sequence if available
    # For now, just verify initialization
    tracker = VehicleTracker()
    print("Tracker ready.")
