# 🚗 Vehicle Vision System: Detection, Tracking & LPR

A high-performance, integrated computer vision pipeline designed for real-time vehicle analytics. This system combines state-of-the-art deep learning models to perform vehicle detection, multi-object tracking, make/model classification, and automatic license plate recognition (ALPR/LPR).

![Pipeline Overview](https://img.shields.io/badge/Pipeline-YOLOv11%20+%20ByteTrack%20+%20EasyOCR-blue)
![Python Version](https://img.shields.io/badge/Python-3.9+-green)

---

## 🌟 Key Features

- **🚀 Real-time Detection**: Powered by **YOLOv11** for rapid identification of cars, trucks, buses, and motorcycles.
- **🛰️ Precision Tracking**: Employs the **ByteTrack** algorithm to maintain consistent vehicle IDs across frames, even through occlusions.
- **🔍 Make/Model Intelligence**: Uses a specialized Vision Transformer (ViT) to classify vehicles by specific make and model.
- **🔢 Automated License Plate Recognition (ALPR)**: Integrated **EasyOCR** engine for high-accuracy text extraction from license plates.
- **🎥 Multi-Input Support**: Seamlessly process local video files, direct URLs, YouTube links, or live webcam/RTSP streams.
- **📏 Optimized Overlays**: Clean, professional video overlays showing only essential information (Make/Model + Plate) in a non-obtrusive font.

---

## 🏗️ System Architecture

The system operates as a sequential pipeline with intelligent frame-skipping to maximize performance:

1.  **Detection Layer**: YOLOv11 analyzes the frame to find bounding boxes for vehicles.
2.  **Tracking Layer**: ByteTrack associates detections across time to assign persistent IDs.
3.  **Processing Queue**:
    *   **Classification**: Every 10th frame, the vehicle crop is passed to the Make/Model classifier.
    *   **OCR**: Every 30th frame, the license plate area is extracted and read by EasyOCR.
4.  **Reporting**: Results are logged to a CSV and visualized with real-time HUD overlays.

---

## 🛠️ Installation

1. **Clone & Setup Environment**:
   ```bash
   git clone <your-repo-url>
   cd vehicle-vision-system
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install Dependencies**:
   ```bash
   pip install ultralytics opencv-python easyocr transformers timm pandas yt-dlp
   ```

---

## 🚀 Usage Guide

### 1. Interactive Pipeline (`run_pipeline.py`)
The most versatile way to process videos. Supports local files, YouTube links, and direct URLs.
```bash
python run_pipeline.py
```
*Follow the on-screen prompts to provide your input and choose output paths.*

### 2. Live Camera Feed (`run_live.py`)
Optimized for real-time webcam or IP camera monitoring.
```bash
python run_live.py
```
*   **Controls**: 
    *   `q`: Quit the application
    *   `s`: Save current session results to CSV

---

## 📊 Output Data Format

The system generates a comprehensive `results.csv` with the following telemetry:

| Column | Description |
| :--- | :--- |
| `frame_id` | Sequence number of the video frame. |
| `vehicle_id` | Persistent ID assigned by the tracker. |
| `vehicle_class` | Category (Car, Truck, Bus, Motorcycle). |
| `make_model` | Identified vehicle brand and model. |
| `license_plate` | Extracted text from the license plate. |
| `confidence` | Combined confidence scores for all predictions. |

---

## 🔬 Technical Stack

*   **Detector**: [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (Pre-trained on COCO)
*   **Tracker**: ByteTrack (Integrated in Ultralytics)
*   **Classifier**: HuggingFace `dima806/car_models_image_detection` (ViT)
*   **OCR**: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
*   **Video Engine**: OpenCV (Open Source Computer Vision Library)

---

## ⚠️ Important Notes/Waymo Dataset

The **Waymo Open Dataset** is excellent for testing detection and tracking, but please note that **license plates are blurred** for privacy. For testing the OCR component, please use unblurred footage or the provided `mock_plate.jpg` in the `data/` directory.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
