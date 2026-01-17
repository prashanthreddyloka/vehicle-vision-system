# Vehicle License Plate Scanner & Classification System

A comprehensive computer vision project for vehicle detection, tracking, counting, classification (make/model), and license plate OCR.

## Features

- **Vehicle Detection**: Uses YOLOv11 to detect cars, trucks, buses, and motorcycles
- **Vehicle Tracking**: ByteTrack algorithm to maintain unique IDs across frames
- **Vehicle Counting**: Tracks total unique vehicles seen
- **Make/Model Classification**: Deep learning classifier for vehicle identification
- **License Plate OCR**: EasyOCR for reading license plate text

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install ultralytics opencv-python easyocr transformers timm pandas
```

## Project Structure

```
distant-quasar/
├── src/
│   ├── detector.py       # Vehicle detection (YOLO)
│   ├── tracker.py        # Vehicle tracking & counting
│   ├── classifier.py     # Make/model classification
│   ├── lpr.py           # License plate OCR
│   └── generate_mock_data.py  # Generate test images
├── data/                # Input/output data
├── models/              # Model weights (auto-downloaded)
├── main.py             # Main pipeline
└── README.md
```

## Usage

### Quick Test

```python
from main import VehicleAnalysisPipeline

pipeline = VehicleAnalysisPipeline()
pipeline.process_video('input.mp4', 'output.mp4', max_frames=100)
pipeline.save_results('results.csv')
```

### Live Camera/Webcam

```bash
python run_live.py
```

Process live video from your webcam or IP camera in real-time. Press 'q' to quit, 's' to save results.

### Individual Components

#### Vehicle Detection
```python
from src.detector import VehicleDetector
import cv2

detector = VehicleDetector()
frame = cv2.imread('car_image.jpg')
detections = detector.detect_vehicles(frame)
```

#### License Plate OCR
```python
from src.lpr import LicensePlateScanner

scanner = LicensePlateScanner()
results = scanner.scan_plate('plate_image.jpg')
```

## Waymo Dataset Notes

⚠️ **Important**: The Waymo Open Dataset blurs license plates and faces for privacy. 
- Use Waymo data for vehicle detection, tracking, and counting
- Use separate unblurred images/videos for license plate OCR testing
- Waymo provides vehicle type labels (Car, Truck, Bus) but not specific make/model

## Output Format

Results are saved as CSV with columns:
- `frame_id`: Frame number
- `vehicle_id`: Unique tracking ID
- `vehicle_class`: Type (car, truck, bus, motorcycle)
- `make_model`: Classified vehicle make/model
- `license_plate`: OCR result (if detected)
- `confidence`: Detection and classification confidence scores

## Performance

- Detection: ~30 FPS on CPU (faster with GPU)
- Classification: ~5-10 FPS per vehicle
- OCR: ~1-2s per plate

## Dependencies

- ultralytics (YOLO)
- opencv-python
- easyocr
- transformers
- timm
- pandas
- numpy

## License

For educational and research purposes.
