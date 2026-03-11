import threading
import time
import cv2
import yaml

from src.tracker import VehicleTracker
from src.space_manager import SpaceManager
from src.match import PlateMatcher
from src.database import VehicleDatabase
from src.grabber import LatestFrameGrabber
from src.crop import crop_bbox
from src.plateDetector import PlateDetector

try:
    from src.lprReader import OCRPlateReader
except ImportError:
    OCRPlateReader = None


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def entry_camera_loop(config, plate_matcher, database, stop_event, display_frame):
    entry_cam_cfg = config.get('entry_camera', {})
    plate_det_cfg = config.get('plate_detection', {})

    try:
        plate_detector = PlateDetector(
            model_path=plate_det_cfg.get('model_path', 'models/plates/plate_detect_model.pt'),
            conf=plate_det_cfg.get('confidence', 0.60),
        )
        print("Plate detector: CPU")
    except Exception as e:
        print(f"Plate detector failed to load: {e}")
        return

    lpr_reader = None
    if OCRPlateReader is not None:
        try:
            lpr_reader = OCRPlateReader()
            print("LPR reader: fast-plate-ocr")
        except Exception as e:
            print(f"LPR reader failed: {e}")

    if lpr_reader is None:
        print("WARNING: No LPR reader configured — plates will not be read")

    try:
        grabber = LatestFrameGrabber(
            source=entry_cam_cfg['source'],
            backend=cv2.CAP_DSHOW,
            width=entry_cam_cfg.get('width', 640),
            height=entry_cam_cfg.get('height', 480),
            warmup_frames=30,
            target_fps=entry_cam_cfg.get('fps', 5),
        )
    except RuntimeError as e:
        print(f"Entry camera unavailable: {e}")
        return

    print("Entry camera running.")

    try:
        while not stop_event.is_set():
            if not grabber.has_new_frame():
                time.sleep(0.02)
                continue

            ok, frame = grabber.read()
            if not ok:
                time.sleep(0.1)
                continue

            plates = plate_detector.detect(frame)

            for p in plates:
                x1, y1, x2, y2 = p.bbox
                plate_crop = crop_bbox(frame, p.bbox)
                if plate_crop is None:
                    continue

                text = ''
                if lpr_reader is not None:
                    try:
                        text = lpr_reader.read(plate_crop)
                    except Exception as e:
                        print(f"LPR read error: {e}")
                        text = ''

                if text:
                    print(f"Plate read at entry: {text} (queue size: {plate_matcher.queue_size()})")
                    plate_matcher.push_plate(text)
                    if database.check_permit(text):
                        print(f"Permit valid for {text}")
                    else:
                        database.record_violation(text, "No valid parking permit")
                        print(f"VIOLATION: No permit for {text}")

                label = text or (lpr_reader.last_plate if lpr_reader else '') or f"{p.conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            display_frame[0] = frame
    except Exception as e:
        print(f"Entry camera thread crashed: {e}")
    finally:
        grabber.release()


_DEMO_VEHICLES = [
    ("7EGN216", "Natalie Wu",   "staff",    "2026-12-31"),
    ("XYZ7890", "Bob Smith",       "monthly",  "2026-06-30"),
    ("DEF4567", "Carol Davis",     "annual",   "2027-01-15"),
    ("GHI8901", "Dan Wilson",      "staff",    "2026-12-31"),
    ("JKL2345", "Eva Martinez",    "monthly",  "2026-09-01"),
    ("MNO6789", "Frank Lee",       "daily",    "2026-03-15"),
    ("PQR3456", "Grace Kim",       "annual",   "2027-06-30"),
    ("STU9012", "Hank Brown",      "staff",    "2026-12-31"),
    ("VWX5678", "Ivy Chen",        "monthly",  "2026-08-01"),
    ("YZA0123", "Jack Taylor",     "annual",   "2027-03-01"),
]


def _seed_demo_data(database):
    """Load demo vehicles and permits if the database is empty."""
    existing = database.conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
    if existing > 0:
        return
    for plate, owner, permit_type, expiry in _DEMO_VEHICLES:
        database.add_vehicle(plate, owner)
        database.add_permit(plate, permit_type, expiry)
    print(f"Seeded {len(_DEMO_VEHICLES)} demo vehicles with permits")


def main():
    try:
        config = load_config('config/settings.yaml')
        print("Config loaded successfully!")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    det_cfg = config['detection']
    headless = config.get('display', {}).get('headless', False)

    tracker = VehicleTracker(
        model_path=det_cfg['model_path'],
        tracker_config=config['tracking']['config'],
        confidence=det_cfg['confidence'],
        fps=config['camera']['fps'],
        process_every_n=det_cfg.get('process_every_n', 1),
        imgsz=det_cfg.get('imgsz', 416),
    )
    print("Vehicle detector: CPU")
    space_manager = SpaceManager(config['spaces']['config'])
    plate_matcher = PlateMatcher(config)

    with VehicleDatabase('data/database.db') as database:
        _seed_demo_data(database)

        stop_event    = threading.Event()
        entry_display = [None]
        entry_thread  = threading.Thread(
            target=entry_camera_loop,
            args=(config, plate_matcher, database, stop_event, entry_display),
            daemon=True
        )
        entry_thread.start()

        cam_cfg = config['camera']
        grabber = LatestFrameGrabber(
            source=cam_cfg['source'],
            backend=cv2.CAP_DSHOW,
            width=cam_cfg.get('width', 640),
            height=cam_cfg.get('height', 480),
            warmup_frames=30,
            target_fps=cam_cfg.get('fps', 10),
        )
        print("Running — press Q to quit.")

        try:
            while True:
                ok, frame = grabber.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                vehicles = tracker.update(frame)
                space_manager.update_occupancy(vehicles)

                for v in vehicles:
                    plate_matcher.try_assign(v['track_id'], v['center'])

                for v in tracker.get_exited():
                    plate      = plate_matcher.get_plate(v['track_id'])
                    space      = space_manager.get_vehicle_space(v['track_id'])
                    track_info = tracker.get_track_info(v['track_id'])
                    entry_time = track_info['first_seen'] if track_info else None
                    database.log_exit(
                        track_id=v['track_id'],
                        plate=plate,
                        space=space,
                        entry_time=entry_time,
                        exit_time=v['exit_time']
                    )
                    print(f"Logged exit — plate: {plate}  space: {space}  duration: {v['duration']:.1f}s")
                    plate_matcher.release(v['track_id'])

                if not headless:
                    frame = space_manager.draw_spaces(frame)

                    cv2.polylines(frame, [plate_matcher.entry_zone], isClosed=True, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, "Entry Zone", (plate_matcher.entry_zone[0][0][0], plate_matcher.entry_zone[0][0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    for v in vehicles:
                        x1, y1, x2, y2 = [int(c) for c in v['bbox']]
                        plate = plate_matcher.get_plate(v['track_id'])
                        label = plate if plate else v['class_name']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                        cv2.putText(frame, f"{label} #{v['track_id']}",
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                    summary = space_manager.get_occupancy_summary()
                    cv2.putText(frame, f"Spaces: {summary['occupied']}/{summary['total']} occupied",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    cv2.imshow('Parking Monitor', frame)
                    if entry_display[0] is not None:
                        cv2.imshow('Entry Camera', entry_display[0])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            stop_event.set()
            grabber.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
