"""
Full system demo: VehicleTracker + SpaceManager + PlateMatcher + VehicleDatabase
Simulates the complete main.py pipeline with mocked YOLO and no real cameras.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
import tempfile
import os
from src.tracker import VehicleTracker
from src.space_manager import SpaceManager
from src.match import PlateMatcher
from src.database import VehicleDatabase

SPACES_FILE   = 'config/parking_spaces.json'
WIDTH, HEIGHT = 1280, 720
FPS           = 30
LOT_W         = 760
PANEL_W       = WIDTH - LOT_W
TOTAL_FRAMES  = 420

DEMO_CONFIG = {
    'entry_zone':  {'polygon': [[0, 600], [200, 600], [200, 720], [0, 720]]},
    'plate_reader': {'poll_interval': 1},
}

ENTRY_POS = (50, 660)

# (track_id, plate, space, plate_frame, enter_frame, park_frame, leave_frame)
SCENARIOS = [
    (1, '7ABC123', 'A1',   0,  25,  80, 180),
    (2, '3XYZ999', 'A2',  50,  75, 130, 230),
    (3, '5DEF456', 'B1',  100, 125, 180, 300),
    (4, '2GHI789', 'A3',  150, 175, 230, 350),
]


def lerp(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))


def get_vehicle_pos(scenario, frame):
    tid, plate, space, plate_f, enter_f, park_f, leave_f = scenario
    if frame < enter_f or frame >= leave_f:
        return None
    if frame < park_f:
        t = (frame - enter_f) / max(park_f - enter_f, 1)
        return lerp(ENTRY_POS, TARGET_CENTERS[space], t)
    return TARGET_CENTERS[space]


def build_mock_result(active_vehicles):
    if not active_vehicles:
        r = MagicMock()
        r.boxes.id = None
        return [r]
    r = MagicMock()
    r.boxes.id.cpu().numpy.return_value   = np.array([v[0] for v in active_vehicles], dtype=float)
    r.boxes.xyxy.cpu().numpy.return_value = np.array([v[1] for v in active_vehicles], dtype=float)
    r.boxes.cls.cpu().numpy.return_value  = np.full(len(active_vehicles), 1.0)
    r.boxes.conf.cpu().numpy.return_value = np.full(len(active_vehicles), 0.9)
    return [r]


def draw_lot(frame, sm, tracker, plate_matcher, entry_log):
    frame = sm.draw_spaces(frame)

    cv2.polylines(frame, [np.array([[0,600],[200,600],[200,720],[0,720]])],
                  True, (0, 220, 255), 2)
    cv2.putText(frame, 'ENTRY ZONE', (5, 595),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

    for v in tracker.active_tracks.values():
        x1, y1, x2, y2 = [int(c) for c in v['bbox']]
        plate = plate_matcher.get_plate(v['track_id'])
        label = plate if plate else v['class_name']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(frame, f"{label} #{v['track_id']}",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

    summary = sm.get_occupancy_summary()
    cv2.putText(frame, f"Spaces: {summary['occupied']}/{summary['total']} occupied",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def draw_panel(frame, plate_matcher, entry_log, exit_log, db_count):
    panel = np.full((HEIGHT, PANEL_W, 3), 20, dtype=np.uint8)
    y = 30

    def text(s, color=(200, 200, 200), scale=0.55, bold=False):
        nonlocal y
        cv2.putText(panel, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2 if bold else 1)
        y += 22

    text('ENTRY CAMERA', (0, 220, 255), scale=0.65, bold=True)
    text(f'Plate queue: {plate_matcher.queue_size()}', (180, 180, 180))
    y += 6

    text('Recent reads:', (180, 180, 180))
    for p in entry_log[-4:]:
        text(f'  {p}', (100, 255, 150))
    y += 10

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (60, 60, 60), 1)
    y += 12

    text('ACTIVE VEHICLES', (0, 200, 255), scale=0.65, bold=True)
    active = plate_matcher.get_all()
    if active:
        for tid, plate in active.items():
            text(f'  #{tid}  {plate}', (255, 200, 0))
    else:
        text('  (none)', (100, 100, 100))
    y += 10

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (60, 60, 60), 1)
    y += 12

    text('EXIT LOG (DB)', (0, 200, 255), scale=0.65, bold=True)
    text(f'Total logged: {db_count}', (180, 180, 180))
    for entry in exit_log[-5:]:
        text(f'  {entry}', (100, 160, 255))

    frame[0:HEIGHT, LOT_W:WIDTH] = panel
    cv2.line(frame, (LOT_W, 0), (LOT_W, HEIGHT), (60, 60, 60), 1)


def main():
    sm = SpaceManager(SPACES_FILE)

    global TARGET_CENTERS
    TARGET_CENTERS = {name: sm.get_space_center(name) for name in sm.spaces}

    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    database = VehicleDatabase(db_path)

    with patch('src.tracker.YOLO') as MockYOLO:
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model
        tracker = VehicleTracker(fps=FPS)

        plate_matcher = PlateMatcher(DEMO_CONFIG)

        entry_log      = []
        exit_log       = []
        db_count       = 0
        track_space    = {}
        frame_num      = 0

        print("Full system demo — press Q to quit, R to restart.")

        while True:
            f = frame_num % TOTAL_FRAMES

            # --- Simulate entry camera plate reads ---
            for scenario in SCENARIOS:
                tid, plate, space, plate_f, enter_f, park_f, leave_f = scenario
                if f == plate_f:
                    plate_matcher.push_plate(plate)
                    entry_log.append(plate)

            # --- Build mocked YOLO detections ---
            active = []
            for scenario in SCENARIOS:
                tid, plate, space, plate_f, enter_f, park_f, leave_f = scenario
                pos = get_vehicle_pos(scenario, f)
                if pos:
                    cx, cy = pos
                    active.append((tid, [cx-40, cy-22, cx+40, cy+22]))

            mock_model.track.return_value = build_mock_result(active)
            blank = np.full((HEIGHT, LOT_W, 3), 35, dtype=np.uint8)
            vehicles = tracker.update(blank)

            # --- Pipe tracker → space manager ---
            occupancy = sm.update_occupancy(vehicles)

            # Update last known space per track
            for name, info in occupancy.items():
                if info['track_id'] is not None:
                    track_space[info['track_id']] = name

            # --- Assign plates to entered vehicles ---
            for v in tracker.get_entered():
                plate_matcher.try_assign(v['track_id'], v['center'])

            # --- Log exited vehicles ---
            for v in tracker.get_exited():
                plate      = plate_matcher.get_plate(v['track_id'])
                space      = track_space.get(v['track_id'])
                track_info = tracker.get_track_info(v['track_id'])
                entry_time = track_info['first_seen'] if track_info else None
                database.log_exit(
                    track_id=v['track_id'],
                    plate=plate,
                    space=space,
                    entry_time=entry_time,
                    exit_time=v['exit_time']
                )
                db_count += 1
                dur = f"{v['duration']:.0f}s"
                exit_log.append(f"{plate or '?'}  {space or '?'}  {dur}")
                plate_matcher.release(v['track_id'])
                track_space.pop(v['track_id'], None)

            # --- Draw ---
            canvas = np.full((HEIGHT, WIDTH, 3), 35, dtype=np.uint8)
            lot = draw_lot(blank, sm, tracker, plate_matcher, entry_log)
            canvas[0:HEIGHT, 0:LOT_W] = lot
            draw_panel(canvas, plate_matcher, entry_log, exit_log, db_count)

            cv2.putText(canvas, f"Frame {f}/{TOTAL_FRAMES}",
                        (LOT_W + 10, HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

            cv2.imshow('Full System Demo', canvas)
            key = cv2.waitKey(1000 // FPS) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                tracker.reset()
                plate_matcher.__init__(DEMO_CONFIG)
                sm.update_occupancy([])
                entry_log.clear()
                exit_log.clear()
                track_space.clear()
                db_count = 0
                frame_num = 0
                continue

            frame_num += 1

    database.close()
    os.unlink(db_path)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
