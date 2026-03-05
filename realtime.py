import time

import cv2

from src.vehicleDetector import VehicleDetector
from src.plateDetector import PlateDetector
from src.crop import crop_bbox
from src.grabber import LatestFrameGrabber


def draw_bbox(img, bbox, label: str = "", thickness: int = 2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    if label:
        cv2.putText(
            img, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
        )


def main():
    VEHICLE_MODEL = "models/vehicle/vehicle_detect_model.pt"
    PLATE_MODEL = "models/plates/plate_detect_model.pt"

    vehicle_detector = VehicleDetector(
        model_path=VEHICLE_MODEL,
        conf=0.70,
        iou=0.5,
        imgsz=640,
    )

    plate_detector = PlateDetector(
        model_path=PLATE_MODEL,
        conf=0.70,
        iou=0.5,
        imgsz=416,   # plate detector can use smaller/larger;
        max_det=10,
    )

    # ---- Real-time capture ----
    # source=0 for webcam; source="sample.mp4" for file; source="rtsp://..." for RTSP
    grabber = LatestFrameGrabber(source=0)

    fps_t0 = time.time()
    fps_count = 0
    fps = 0.0

    # Performance caps
    MAX_VEHICLES_PER_FRAME = 5
    MIN_VEHICLE_AREA = 8000  # skip tiny far vehicles
    SHOW_PLATE_CROPS = True  # set False if it slows you down

    try:
        while True:
            ok, frame = grabber.read()
            if not ok or frame is None:
                continue

            # Optional: downscale the displayed frame (does NOT affect detection unless you detect on the resized frame)
            # For simplicity, detect on the frame as-is.
            H, W = frame.shape[:2]

            # ---- Vehicle detection on full frame ----
            vehicles = vehicle_detector.detect(frame)

            # Filter and cap vehicles (biggest first)
            vehicles_sorted = sorted(
                vehicles,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                reverse=True,
            )

            chosen = []
            for v in vehicles_sorted:
                x1, y1, x2, y2 = v.bbox
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_VEHICLE_AREA:
                    continue
                chosen.append(v)
                if len(chosen) >= MAX_VEHICLES_PER_FRAME:
                    break

            # ---- For each vehicle: crop car, detect plates in car crop ----
            plate_crops = []
            for v in chosen:
                car_crop = crop_bbox(frame, v.bbox)
                if car_crop is None:
                    continue

                plates = plate_detector.detect(car_crop)

                # Draw vehicle bbox on full frame
                draw_bbox(frame, v.bbox, label=f"veh {v.cls} {v.conf:.2f}")

                # Draw plate bboxes on the car crop, and optionally collect crops
                for p in plates:
                    plate_crop = crop_bbox(car_crop, p.bbox)
                    if plate_crop is None:
                        continue
                    plate_crops.append(plate_crop)

                    # If you want to draw plate boxes on the full frame, convert ROI coords -> full coords
                    vx1, vy1, _, _ = v.bbox
                    px1, py1, px2, py2 = p.bbox
                    full_plate_bbox = (vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2)
                    draw_bbox(frame, full_plate_bbox, label=f"plate {p.conf:.2f}", thickness=2)

            # ---- FPS counter ----
            fps_count += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_count / (now - fps_t0)
                fps_t0 = now
                fps_count = 0

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )

            # ---- Show main window ----
            cv2.imshow("Parking Vision - Real-time Test", frame)

            # Optional: show latest plate crops (can be heavy if many)
            if SHOW_PLATE_CROPS and plate_crops:
                # show up to 4 crops
                for i, crop in enumerate(plate_crops[:4]):
                    cv2.imshow(f"PlateCrop{i}", crop)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break

    finally:
        grabber.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()