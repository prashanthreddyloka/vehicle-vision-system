"""
Generate test images with real vehicle content using AI image generation,
or create a comprehensive test with the mock plate image.
"""
import cv2
import numpy as np
import os

def test_with_mock_image():
    """Test pipeline on the existing mock plate image."""
    from src.detector import VehicleDetector
    from src.classifier import VehicleMakeModelClassifier
    from src.lpr import LicensePlateScanner
    
    print("=" * 60)
    print("Testing Individual Components")
    print("=" * 60)
    
    mock_image_path = "data/mock_plate.jpg"
    
    if not os.path.exists(mock_image_path):
        print(f"Error: {mock_image_path} not found!")
        return
    
    img = cv2.imread(mock_image_path)
    print(f"\nLoaded test image: {mock_image_path}")
    print(f"Image shape: {img.shape}")
    
    # Test 1: Vehicle Detection
    print("\n" + "-" * 60)
    print("Test 1: Vehicle Detection (YOLO)")
    print("-" * 60)
    detector = VehicleDetector()
    detections = detector.detect_vehicles(img)
    print(f"Vehicles detected: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class']} - confidence: {det['confidence']:.2f}")
    
    # Test 2: License Plate OCR
    print("\n" + "-" * 60)
    print("Test 2: License Plate OCR (EasyOCR)")
    print("-" * 60)
    scanner = LicensePlateScanner()
    plates = scanner.scan_plate(mock_image_path)
    print(f"License plates detected: {len(plates)}")
    for i, plate in enumerate(plates):
        print(f"  {i+1}. Text: '{plate['text']}' - confidence: {plate['confidence']:.2f}")
    
    # Test 3: Vehicle Classification (if vehicles were detected)
    if detections:
        print("\n" + "-" * 60)
        print("Test 3: Vehicle Make/Model Classification")
        print("-" * 60)
        classifier = VehicleMakeModelClassifier()
        
        for i, det in enumerate(detections[:3]):  # Test on first 3 detections
            x1, y1, x2, y2 = map(int, det['bbox'])
            crop = img[y1:y2, x1:x2]
            
            if crop.size > 0:
                result = classifier.classify(crop)
                print(f"  Vehicle {i+1}: {result['make_model']} - confidence: {result['confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print("Component Testing Complete!")
    print("=" * 60)
    
    # Create visualization
    if detections:
        output_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(output_img, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = "data/test_result.jpg"
        cv2.imwrite(output_path, output_img)
        print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    test_with_mock_image()
