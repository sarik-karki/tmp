from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from src.tracker import VehicleTracker

WIDTH, HEIGHT = 1280, 720
FPS = 30


# --- Fake vehicle paths ---
# Each entry: (track_id, class_id, start_frame, end_frame, start_pos, end_pos)
VEHICLES = [
    (1, 1, 0,  180, (0,   200), (1280, 200)),   # car,        left → right
    (2, 0, 20, 200, (1280, 400), (0,   400)),   # bus,        right → left
    (3, 2, 60, 180, (640,  0),  (640,  720)),   # motorcycle, top → bottom
    (4, 1, 100,220, (0,   580), (1280, 580)),   # car,        left → right
    (5, 3, 40, 160, (200,  0),  (200,  720)),   # shuttle,    top → bottom
]
TOTAL_FRAMES = 250


def lerp_pos(start, end, t):
    return (int(start[0] + (end[0] - start[0]) * t),
            int(start[1] + (end[1] - start[1]) * t))


def make_fake_result(active):
    """Build a mocked YOLO result for the current active vehicles."""
    if not active:
        result = MagicMock()
        result.boxes.id = None
        return [result]

    track_ids = np.array([v['track_id'] for v in active], dtype=float)
    boxes     = np.array([v['bbox'] for v in active], dtype=float)
    class_ids = np.array([v['class_id'] for v in active], dtype=float)
    confs     = np.array([0.9] * len(active), dtype=float)

    result = MagicMock()
    result.boxes.id.cpu().numpy.return_value = track_ids
    result.boxes.xyxy.cpu().numpy.return_value = boxes
    result.boxes.cls.cpu().numpy.return_value = class_ids
    result.boxes.conf.cpu().numpy.return_value = confs
    return [result]


def get_active_at(frame_num):
    active = []
    for tid, cls, f_start, f_end, pos_start, pos_end in VEHICLES:
        if f_start <= frame_num < f_end:
            t = (frame_num - f_start) / (f_end - f_start)
            cx, cy = lerp_pos(pos_start, pos_end, t)
            w, h = (120, 60) if cls == 0 else (70, 40)
            active.append({
                'track_id': tid,
                'class_id': cls,
                'bbox': [cx - w//2, cy - h//2, cx + w//2, cy + h//2],
            })
    return active


def draw_vehicle(frame, bbox, track_id, class_name):
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    label = f"{class_name} #{track_id}"
    cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)


def draw_event(frame, events, y_offset, color, prefix):
    for i, e in enumerate(events[-3:]):  # show last 3
        text = f"{prefix}: {e['class_name']} #{e['track_id']}"
        cv2.putText(frame, text, (10, y_offset + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def main():
    with patch('src.tracker.YOLO') as MockYOLO:
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model

        tracker = VehicleTracker(fps=FPS)

        entered_log = []
        exited_log  = []
        frame_num   = 0
        loop        = True

        print("Tracker demo — press Q to quit, R to restart.")

        while loop:
            active = get_active_at(frame_num % TOTAL_FRAMES)
            mock_model.track.return_value = make_fake_result(active)

            blank_frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
            vehicles = tracker.update(blank_frame)

            entered_log.extend(tracker.get_entered())
            exited_log.extend(tracker.get_exited())

            # Draw vehicles
            for v in vehicles:
                draw_vehicle(blank_frame, v['bbox'], v['track_id'], v['class_name'])

            # HUD
            cv2.putText(blank_frame, f"Frame: {frame_num % TOTAL_FRAMES:>4}   Active: {tracker.get_active_count()}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            draw_event(blank_frame, entered_log, 80,  (0, 255, 100), "ENTERED")
            draw_event(blank_frame, exited_log,  160, (0, 100, 255), "EXITED ")

            cv2.imshow('Tracker Demo', blank_frame)
            key = cv2.waitKey(1000 // FPS) & 0xFF
            if key == ord('q') or key == 27:
                loop = False
            elif key == ord('r'):
                tracker.reset()
                entered_log.clear()
                exited_log.clear()
                frame_num = 0
                continue

            frame_num += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
