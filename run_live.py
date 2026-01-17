"""
Live camera/webcam processing for vehicle detection and license plate recognition.
Press 'q' to quit, 's' to save current results.
"""

import cv2
import os
from datetime import datetime
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.classifier import VehicleMakeModelClassifier
from src.lpr import LicensePlateScanner
import pandas as pd

class LiveVehicleAnalysis:
    def __init__(self):
        print("Initializing Live Vehicle Analysis...")
        self.detector = VehicleDetector()
        self.tracker = VehicleTracker()
        self.classifier = VehicleMakeModelClassifier()
        self.lpr_scanner = LicensePlateScanner()
        self.results = []
        self.frame_count = 0
        print("✅ Live pipeline ready!\n")

    def process_frame(self, frame):
        """Process a single frame with all components."""
        self.frame_count += 1
        
        # Track vehicles
        detections, total_count = self.tracker.track_and_count(frame)
        
        # Process each detected vehicle
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            vehicle_crop = frame[y1:y2, x1:x2]
            
            # Only do expensive operations every N frames or for new vehicles
            skip_expensive = self.frame_count % 10 != 0  # Process every 10th frame
            
            make_model = "N/A"
            make_model_conf = 0.0
            plate_text = "N/A"
            plate_conf = 0.0
            
            if not skip_expensive and vehicle_crop.size > 0:
                # Classify make/model
                try:
                    result = self.classifier.classify(vehicle_crop)
                    make_model = result['make_model']
                    make_model_conf = result['confidence']
                except Exception as e:
                    print(f"Classification error: {e}")
                
                # OCR license plate (slower, do less frequently)
                if self.frame_count % 30 == 0:  # Every 30 frames
                    try:
                        temp_path = f"data/temp_live_{detection['id']}.jpg"
                        cv2.imwrite(temp_path, vehicle_crop)
                        plates = self.lpr_scanner.scan_plate(temp_path)
                        if plates:
                            best = max(plates, key=lambda p: p['confidence'])
                            plate_text = best['text']
                            plate_conf = best['confidence']
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception as e:
                        print(f"OCR error: {e}")
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw license plate if detected (at bottom of bbox)
            if plate_text != "N/A":
                cv2.putText(frame, f"Plate: {plate_text}", (x1, y2+12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            # Draw make/model if detected (at top of bbox)
            display_label = make_model if make_model != "N/A" else detection['class']
            cv2.putText(frame, display_label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            # Store result
            self.results.append({
                "timestamp": datetime.now().isoformat(),
                "frame_id": self.frame_count,
                "vehicle_id": detection['id'],
                "vehicle_class": detection['class'],
                "detection_confidence": detection['confidence'],
                "make_model": make_model,
                "make_model_confidence": make_model_conf,
                "license_plate": plate_text,
                "plate_confidence": plate_conf,
                "bbox": detection['bbox']
            })
        
        # Draw info overlay (smaller)
        cv2.putText(frame, f"Total Vehicles: {total_count}", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(frame, "q:quit | s:save", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return frame, total_count

    def save_results(self, filename="data/live_results.csv"):
        """Save accumulated results to CSV."""
        if not self.results:
            print("No results to save.")
            return
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"✅ Results saved to: {filename}")

    def run_live(self, camera_source=0):
        """
        Run live processing from camera.
        
        Args:
            camera_source: 0 for default webcam, or a stream URL like:
                          'rtsp://...' for IP camera
                          'http://...' for HTTP stream
        """
        print("=" * 70)
        print("Starting Live Camera Feed...")
        print("=" * 70)
        print(f"Camera source: {camera_source}")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save results")
        print("=" * 70)
        print()
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera source: {camera_source}")
            return
        
        # Set camera properties (optional)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Camera opened successfully!")
        print("   Processing live feed...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, total_count = self.process_frame(frame)
                
                # Display
                cv2.imshow('Live Vehicle Analysis', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠️  Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/live_results_{timestamp}.csv"
                    self.save_results(filename)
                    
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Save final results
            print("\n" + "=" * 70)
            self.save_results()
            
            # Summary
            if self.results:
                unique_vehicles = len(set([r['vehicle_id'] for r in self.results]))
                plates_detected = sum(1 for r in self.results if r['license_plate'] != "N/A")
                print(f"\n📈 Session Summary:")
                print(f"   - Total frames processed: {self.frame_count}")
                print(f"   - Total detections logged: {len(self.results)}")
                print(f"   - Unique vehicles: {unique_vehicles}")
                print(f"   - License plates read: {plates_detected}")
            print("=" * 70)

def main():
    import sys
    
    print("\n" + "=" * 70)
    print("  Live Vehicle Detection & License Plate Recognition")
    print("=" * 70)
    print()
    
    # Ask for camera source
    print("Select camera source:")
    print("  0 - Default webcam")
    print("  1 - Secondary camera")
    print("  Or enter stream URL (rtsp://... or http://...)")
    print()
    
    source_input = input("Camera source (press Enter for default webcam): ").strip()
    
    if not source_input:
        camera_source = 0
    elif source_input.isdigit():
        camera_source = int(source_input)
    else:
        camera_source = source_input
    
    # Run live analysis
    analyzer = LiveVehicleAnalysis()
    analyzer.run_live(camera_source)

if __name__ == "__main__":
    main()
