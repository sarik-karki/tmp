"""
Single-Webcam Hybrid Demo: Splits one camera feed into two virtual cameras.
Tests the full pipeline (Entrance + Overhead) using just your laptop webcam.
"""
import threading
import time
import cv2
import yaml
import numpy as np

from src.tracker import VehicleTracker
from src.space_manager import SpaceManager
from src.match import PlateMatcher
from src.database import VehicleDatabase
from src.crop import crop_bbox
from src.plateDetector import PlateDetector
from main import _seed_demo_data

try:
    from src.lprReader import OCRPlateReader
except ImportError:
    OCRPlateReader = None

def main():
    try:
        with open('config/settings.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {'detection': {'model_path': 'models/vehicle/vehicle_detect_model.pt', 'confidence': 0.5},
                  'plate_detection': {'model_path': 'models/plates/plate_detect_model.pt', 'confidence': 0.5},
                  'tracking': {'config': 'config/bytetrack.yaml'},
                  'spaces': {'config': 'config/parking_spaces.json'}}

    # Initialize models
    plate_detector = PlateDetector(config['plate_detection']['model_path'])
    tracker = VehicleTracker(model_path=config['detection']['model_path'])
    space_manager = SpaceManager(config['spaces']['config'])
    
    # Override entry zone for this split-screen demo
    # We'll treat the right half of the screen as the lot
    # Entry zone covers the overhead view (right half, but coordinates are 0-640 after crop)
    config['entry_zone'] = {'polygon': [[0, 0], [640, 0], [640, 720], [0, 720]]}
    plate_matcher = PlateMatcher(config)

    lpr_reader = OCRPlateReader() if OCRPlateReader else None

    # Open the webcam (index 0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Single-Webcam Hybrid Demo ---")
    print("Left Half: Simulated Entrance (Plate Detection)")
    print("Right Half: Simulated Overhead (Tracking & Spaces)")
    print("Press 'Q' to quit.")

    with VehicleDatabase('data/demo_hybrid.db') as database:
        _seed_demo_data(database)

        while True:
            ok, full_frame = cap.read()
            if not ok: break

            # Split the frame
            # Entrance view = Left half, Overhead view = Right half
            h, w = full_frame.shape[:2]
            mid = w // 2
            entrance_view = full_frame[:, :mid].copy()
            overhead_view = full_frame[:, mid:].copy()

            # 1. Process Entrance (Left Side)
            plates = plate_detector.detect(entrance_view)
            for p in plates:
                plate_crop = crop_bbox(entrance_view, p.bbox)
                text = lpr_reader.read(plate_crop) if lpr_reader and plate_crop is not None else ""
                if text:
                    print(f"Plate read: {text}")
                    plate_matcher.push_plate(text)
                
                x1, y1, x2, y2 = p.bbox
                label = text or (lpr_reader.last_plate if lpr_reader else '') or f"{p.conf:.2f}"
                cv2.rectangle(entrance_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(entrance_view, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 2. Process Overhead (Right Side)
            # Adjust coordinates for the tracker because it's only seeing the right half
            vehicles = tracker.update(overhead_view)
            space_manager.update_occupancy(vehicles)

            for v in vehicles:
                plate_matcher.try_assign(v['track_id'], v['center'])
                
                # Draw
                x1, y1, x2, y2 = [int(c) for c in v['bbox']]
                plate = plate_matcher.get_plate(v['track_id'])
                label = plate if plate else v['class_name']
                cv2.rectangle(overhead_view, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(overhead_view, f"{label} #{v['track_id']}", (x1, max(0, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)

            # 3. Combine for Display
            overhead_view = space_manager.draw_spaces(overhead_view)
            
            # Draw dividers and labels
            display = np.hstack((entrance_view, overhead_view))
            cv2.line(display, (mid, 0), (mid, h), (255, 255, 255), 2)
            cv2.putText(display, "ENTRANCE (Plate Cam)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "LOT (Overhead Cam)", (mid + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            cv2.imshow("Parking Monitor - Hybrid Demo", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
