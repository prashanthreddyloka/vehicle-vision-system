"""
Main pipeline for Vehicle License Plate Scanner and Classification System.

This script integrates:
1. Vehicle Detection (YOLO)
2. Vehicle Tracking (ByteTrack)
3. Vehicle Make/Model Classification (HuggingFace ViT)
4. License Plate OCR (EasyOCR)
"""

import cv2
import os
import pandas as pd
from datetime import datetime
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.classifier import VehicleMakeModelClassifier
from src.lpr import LicensePlateScanner

class VehicleAnalysisPipeline:
    def __init__(self):
        print("Initializing Vehicle Analysis Pipeline...")
        self.detector = VehicleDetector()
        self.tracker = VehicleTracker()
        self.classifier = VehicleMakeModelClassifier()
        self.lpr_scanner = LicensePlateScanner()
        self.results = []
        print("Pipeline ready.\n")

    def process_frame(self, frame, frame_id):
        """Process a single frame through the complete pipeline."""
        # Step 1: Track vehicles
        detections, total_count = self.tracker.track_and_count(frame)
        
        # Store enhanced detection data for display
        enhanced_detections = []
        
        for detection in detections:
            # Extract vehicle crop
            x1, y1, x2, y2 = map(int, detection['bbox'])
            vehicle_crop = frame[y1:y2, x1:x2]
            
            # Step 2: Classify vehicle make/model
            make_model_result = self.classifier.classify(vehicle_crop)
            
            # Step 3: Try to detect license plate in the crop
            plate_text = None
            plate_conf = 0.0
            
            # Save crop temporarily for OCR
            temp_crop_path = f"data/temp_vehicle_{detection['id']}.jpg"
            cv2.imwrite(temp_crop_path, vehicle_crop)
            
            plates = self.lpr_scanner.scan_plate(temp_crop_path)
            if plates:
                # Get the highest confidence plate
                best_plate = max(plates, key=lambda p: p['confidence'])
                plate_text = best_plate['text']
                plate_conf = best_plate['confidence']
            
            # Clean up temp file
            if os.path.exists(temp_crop_path):
                os.remove(temp_crop_path)
            
            # Store results
            self.results.append({
                "frame_id": frame_id,
                "vehicle_id": detection['id'],
                "vehicle_class": detection['class'],
                "detection_confidence": detection['confidence'],
                "make_model": make_model_result['make_model'],
                "make_model_confidence": make_model_result['confidence'],
                "license_plate": plate_text if plate_text else "N/A",
                "plate_confidence": plate_conf,
                "bbox": detection['bbox']
            })
            
            # Add enhanced info for display
            enhanced_detections.append({
                **detection,
                'make_model': make_model_result['make_model'],
                'license_plate': plate_text if plate_text else "N/A"
            })
        
        return total_count, enhanced_detections


    def process_video(self, video_path, output_video_path=None, max_frames=None):
        """Process a video file through the pipeline."""
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            return
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer for output
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_id >= max_frames:
                break
            
            total_count, detections = self.process_frame(frame, frame_id)
            
            # Draw overlays
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Display make/model and license plate
                make_model = det.get('make_model', 'Unknown')
                plate = det.get('license_plate', 'N/A')
                
                # Draw make/model
                cv2.putText(frame, make_model, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                # Draw license plate if detected
                if plate != "N/A":
                    cv2.putText(frame, f"Plate: {plate}", (x1, y2+12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            # Display total count
            cv2.putText(frame, f"Total Vehicles: {total_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            if out:
                out.write(frame)
            
            frame_id += 1
            if frame_id % 30 == 0:
                print(f"Processed {frame_id} frames, total vehicles tracked: {total_count}")
        
        cap.release()
        if out:
            out.release()
        
        print(f"\nProcessing complete. Total frames: {frame_id}")
        print(f"Total unique vehicles tracked: {total_count}")

    def save_results(self, csv_path="data/results.csv"):
        """Save results to CSV."""
        if not self.results:
            print("No results to save.")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    # Example usage
    pipeline = VehicleAnalysisPipeline()
    
    # For testing, we can just verify initialization
    print("Pipeline initialized successfully!")
    print("To use: pipeline.process_video('path/to/video.mp4', 'output.mp4')")
