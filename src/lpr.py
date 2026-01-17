import cv2
import easyocr
import os

class LicensePlateScanner:
    def __init__(self):
        # Initialize EasyOCR (gpu=False for broad compatibility)
        # verbose=False to avoid UnicodeEncodeError in progress bar
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("LPR Scanner (EasyOCR) initialized.")

    def scan_plate(self, image_path):
        """Scans the image for text (specifically license plates)."""
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found.")
            return []

        # Load image with OpenCV first
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return []

        # readtext returns a list of tuples: (bounding box, text, confidence)
        # Pass the image array instead of the path
        results = self.reader.readtext(image)
        
        plates = []
        for (bbox, text, prob) in results:
            plates.append({"text": text, "confidence": prob})
        
        return plates

if __name__ == "__main__":
    scanner = LicensePlateScanner()
    sample_img = "data/mock_plate.jpg"
    results = scanner.scan_plate(sample_img)
    
    print(f"\nScanning result for {sample_img}:")
    if results:
        for r in results:
            print(f"Detected Text: {r['text']} (Confidence: {r['confidence']:.2f})")
    else:
        print("No text detected.")
