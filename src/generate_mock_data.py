import cv2
import os
import numpy as np

def generate_mock_lpr_data():
    """Outputs a mock frame with a dummy license plate for testing OCR."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50) # Dark gray background
    
    # Draw a "car" (simple rectangle)
    cv2.rectangle(frame, (400, 300), (880, 600), (150, 150, 150), -1)
    
    # Draw a license plate area
    plate_coords = (590, 530, 200, 50) # x, y, w, h
    cv2.rectangle(frame, (plate_coords[0], plate_coords[1]), 
                  (plate_coords[0]+plate_coords[2], plate_coords[1]+plate_coords[3]), (255, 255, 255), -1)
    
    # Add text to the plate
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "ABC-1234", (plate_coords[0]+10, plate_coords[1]+40), font, 1.2, (0, 0, 0), 3)
    
    output_path = "data/mock_plate.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Mock plate image saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_mock_lpr_data()
