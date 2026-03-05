"""
Combined demo: VehicleTracker output feeds directly into SpaceManager —
the same pipeline as main.py, with YOLO mocked and no real camera.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from src.tracker import VehicleTracker
from src.space_manager import SpaceManager

SPACES_FILE = 'config/parking_spaces.json'
WIDTH, HEIGHT = 1280, 720
FPS = 30
TOTAL_FRAMES = 300

# (track_id, class_id, frame_start, frame_end, origin, target_space_center)
# origin = starting position outside the spaces (above the lot)
VEHICLE_PATHS = [
    (1, 1, 0,   120, (160, 50),  (160, 375)),  # car   → A1
    (2, 1, 20,  140, (300, 50),  (300, 375)),  # car   → A2
    (3, 2, 50,  160, (440, 50),  (440, 375)),  # moto  → A3
    (4, 0, 80,  200, (160, 720), (160, 545)),  # bus   → B1 (from bottom)
    (5, 1, 100, 220, (300, 720), (300, 545)),  # car   → B2 (from bottom)
    (6, 3, 120, 240, (440, 720), (440, 545)),  # shuttle→B3 (from bottom)
]


def lerp(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))


def get_active_at(frame_num):
    active = []
    for tid, cls, f_start, f_end, origin, target in VEHICLE_PATHS:
        if f_start <= frame_num < f_end:
            t = (frame_num - f_start) / (f_end - f_start)
            cx, cy = lerp(origin, target, t)
            w, h = (110, 55) if cls == 0 else (70, 40)
            active.append({
                'track_id': tid,
                'class_id': cls,
                'bbox': [cx - w//2, cy - h//2, cx + w//2, cy + h//2],
            })
    return active


def build_mock_result(active):
    if not active:
        r = MagicMock()
        r.boxes.id = None
        return [r]

    r = MagicMock()
    r.boxes.id.cpu().numpy.return_value   = np.array([v['track_id'] for v in active], dtype=float)
    r.boxes.xyxy.cpu().numpy.return_value = np.array([v['bbox']     for v in active], dtype=float)
    r.boxes.cls.cpu().numpy.return_value  = np.array([v['class_id'] for v in active], dtype=float)
    r.boxes.conf.cpu().numpy.return_value = np.full(len(active), 0.9)
    return [r]


def draw_vehicle(frame, v):
    x1, y1, x2, y2 = [int(c) for c in v['bbox']]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.putText(frame, f"{v['class_name']} #{v['track_id']}",
                (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def main():
    sm = SpaceManager(SPACES_FILE)

    with patch('src.tracker.YOLO') as MockYOLO:
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model
        tracker = VehicleTracker(fps=FPS)

        frame_num = 0
        print("Combined demo — press Q to quit, R to restart.")

        while True:
            f = frame_num % TOTAL_FRAMES

            # --- Feed mocked detections into real tracker ---
            active = get_active_at(f)
            mock_model.track.return_value = build_mock_result(active)
            blank = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
            vehicles = tracker.update(blank)          # real tracker logic

            # --- Pipe tracker output directly into space manager ---
            sm.update_occupancy(vehicles)             # real space manager logic

            # --- Draw ---
            frame = sm.draw_spaces(blank)             # spaces: green/red

            for v in vehicles:
                draw_vehicle(frame, v)

            summary = sm.get_occupancy_summary()
            cv2.putText(frame,
                        f"Tracked: {tracker.get_active_count()}   "
                        f"Spaces: {summary['occupied']}/{summary['total']} occupied",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Entered / exited events
            for i, v in enumerate(tracker.get_entered()):
                cv2.putText(frame, f"ENTERED: {v['class_name']} #{v['track_id']}",
                            (10, 75 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
            for i, v in enumerate(tracker.get_exited()):
                cv2.putText(frame, f"EXITED:  {v['class_name']} #{v['track_id']}",
                            (10, 140 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)

            cv2.putText(frame, f"Frame {f}/{TOTAL_FRAMES}",
                        (WIDTH - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

            cv2.imshow('Combined Demo — Tracker + SpaceManager', frame)
            key = cv2.waitKey(1000 // FPS) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                tracker.reset()
                sm.update_occupancy([])
                frame_num = 0
                continue

            frame_num += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
