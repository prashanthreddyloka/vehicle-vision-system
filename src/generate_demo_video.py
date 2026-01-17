"""
Demo script to generate a sample video with cars and test the complete pipeline.
"""
import cv2
import numpy as np
import os

def generate_demo_video(output_path="data/demo_video.mp4", num_frames=100):
    """Generate a simple demo video with moving rectangles simulating cars."""
    width, height = 1280, 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Simulate 3 cars moving across the screen
    cars = [
        {"x": 0, "y": 200, "speed": 8, "color": (150, 150, 150)},
        {"x": 0, "y": 400, "speed": 5, "color": (100, 100, 150)},
        {"x": 200, "y": 300, "speed": 6, "color": (120, 140, 120)},
    ]
    
    for frame_num in range(num_frames):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray
        
        # Draw road
        cv2.rectangle(frame, (0, 150), (width, 550), (70, 70, 70), -1)
        
        # Draw lane markings
        for i in range(0, width, 100):
            cv2.rectangle(frame, (i, 345), (i+50, 355), (200, 200, 200), -1)
        
        # Draw and move cars
        for car in cars:
            # Draw car body
            cv2.rectangle(frame, (car['x'], car['y']), 
                         (car['x']+120, car['y']+60), car['color'], -1)
            
            # Draw "license plate"
            cv2.rectangle(frame, (car['x']+30, car['y']+45), 
                         (car['x']+90, car['y']+55), (255, 255, 255), -1)
            
            # Move car
            car['x'] += car['speed']
            if car['x'] > width:
                car['x'] = -120
        
        out.write(frame)
    
    out.release()
    print(f"Demo video created: {output_path}")
    return output_path

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    video_path = generate_demo_video()
    print(f"\nDemo video ready at: {video_path}")
    print("Run: python -c \"from main import VehicleAnalysisPipeline; p = VehicleAnalysisPipeline(); p.process_video('data/demo_video.mp4', 'data/output.mp4')\"")
