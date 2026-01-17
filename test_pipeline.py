"""
Test script to run the complete Vehicle Analysis Pipeline.
"""

from main import VehicleAnalysisPipeline
import os

def test_pipeline():
    print("=" * 60)
    print("Vehicle Analysis Pipeline - Verification Test")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = VehicleAnalysisPipeline()
    
    # Process demo video (limiting to 100 frames for quick testing)
    video_input = "data/demo_video.mp4"
    video_output = "data/processed_demo.mp4"
    csv_output = "data/verification_results.csv"
    
    if not os.path.exists(video_input):
        print(f"Error: {video_input} not found!")
        return
    
    print(f"\nProcessing video: {video_input}")
    print(f"Output will be saved to: {video_output}")
    print(f"Results will be saved to: {csv_output}")
    print("\nProcessing (limited to 100 frames for verification)...\n")
    
    # Run pipeline
    pipeline.process_video(video_input, video_output, max_frames=100)
    
    # Save results
    pipeline.save_results(csv_output)
    
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)
    print(f"\nCheck outputs:")
    print(f"  - Processed video: {video_output}")
    print(f"  - Results CSV: {csv_output}")
    
    # Display summary of results
    if pipeline.results:
        print(f"\nSummary:")
        print(f"  Total detections logged: {len(pipeline.results)}")
        unique_vehicles = len(set([r['vehicle_id'] for r in pipeline.results]))
        print(f"  Unique vehicles tracked: {unique_vehicles}")
        
        # Count vehicles with plates detected
        plates_detected = sum(1 for r in pipeline.results if r['license_plate'] != "N/A")
        print(f"  License plates detected: {plates_detected}")
        
        # Show sample results
        print(f"\nSample results (first 3):")
        for i, result in enumerate(pipeline.results[:3]):
            print(f"\n  Detection {i+1}:")
            print(f"    Frame: {result['frame_id']}")
            print(f"    Vehicle ID: {result['vehicle_id']}")
            print(f"    Class: {result['vehicle_class']}")
            print(f"    Make/Model: {result['make_model']} (conf: {result['make_model_confidence']:.2f})")
            print(f"    License Plate: {result['license_plate']} (conf: {result['plate_confidence']:.2f})")

if __name__ == "__main__":
    test_pipeline()
