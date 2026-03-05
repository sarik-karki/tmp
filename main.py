import threading
import time
import yaml
import os
import re
import cv2
import requests
from dotenv import load_dotenv

from src.tracker import VehicleTracker
from src.space_manager import SpaceManager
from src.match import PlateMatcher
from src.database import VehicleDatabase
from src.grabber import LatestFrameGrabber
from src.plateDetector import PlateDetector
from src.crop import crop_bbox


def load_config(path):
    with open(path, 'r') as f:
        content = f.read()
    expanded_content = os.path.expandvars(content)
    missing_vars = re.findall(r'\${?([\w\.\-]+)}?', expanded_content)
    if missing_vars:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(set(missing_vars))}. "
            "Please check your .env file."
        )
    return yaml.safe_load(expanded_content)


def read_plate_from_api(api_url, plate_crop):
    _, img_encoded = cv2.imencode('.jpg', plate_crop)
    try:
        response = requests.post(
            api_url,
            files={'image': ('plate.jpg', img_encoded.tobytes(), 'image/jpeg')},
            timeout=3
        )
        if response.ok:
            return response.json().get('plate', '').upper().strip()
    except requests.RequestException:
        pass
    return ''


def entry_camera_loop(config, plate_matcher, stop_event):
    plate_detector = PlateDetector(
        model_path='models/plates/plate_detect_model.pt',
        conf=0.60,
    )
    api_url       = config.get('plate_reader', {}).get('api_url', '')
    poll_interval = config.get('plate_reader', {}).get('poll_interval', 1)

    try:
        grabber = LatestFrameGrabber(
            source=config['entry_camera']['source'],
            backend=cv2.CAP_V4L2,
            width=1280,
            height=720,
            warmup_frames=30,
        )
    except RuntimeError as e:
        print(f"Entry camera unavailable: {e}")
        return

    print("Entry camera running.")

    while not stop_event.is_set():
        ok, frame = grabber.read()
        if not ok:
            time.sleep(0.1)
            continue
        plates = plate_detector.detect(frame)
        for p in plates:
            plate_crop = crop_bbox(frame, p.bbox)
            if plate_crop is None:
                continue
            text = read_plate_from_api(api_url, plate_crop)
            if text:
                print(f"Plate read at entry: {text}")
                plate_matcher.push_plate(text)
        time.sleep(poll_interval)

    grabber.release()


def main():
    load_dotenv()

    try:
        config = load_config('config/settings.yaml')
        print("Config loaded successfully!")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    tracker       = VehicleTracker(
        model_path=config['detection']['model_path'],
        tracker_config=config['tracking']['config'],
        confidence=config['detection']['confidence'],
        fps=config['camera']['fps']
    )
    space_manager = SpaceManager(config['spaces']['config'])
    plate_matcher = PlateMatcher(config)
    database      = VehicleDatabase('data/database.db')

    stop_event   = threading.Event()
    entry_thread = threading.Thread(
        target=entry_camera_loop,
        args=(config, plate_matcher, stop_event),
        daemon=True
    )
    entry_thread.start()

    grabber = LatestFrameGrabber(
        source=config['camera']['source'],
        backend=cv2.CAP_V4L2,
        width=1280,
        height=720,
        warmup_frames=30,
    )
    print("Running — press Q to quit.")

    TRACK_W, TRACK_H = 640, 360  # downscale for YOLO inference

    try:
        while True:
            ok, frame = grabber.read()
            if not ok:
                time.sleep(0.01)
                continue

            H, W = frame.shape[:2]
            track_frame = cv2.resize(frame, (TRACK_W, TRACK_H))
            vehicles = tracker.update(track_frame)

            # Scale bboxes and centers back to full resolution for drawing/space checks
            scale_x = W / TRACK_W
            scale_y = H / TRACK_H
            for v in vehicles:
                x1, y1, x2, y2 = v['bbox']
                v['bbox'] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                cx, cy = v['center']
                v['center'] = (int(cx * scale_x), int(cy * scale_y))

            space_manager.update_occupancy(vehicles)

            for v in tracker.get_entered():
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

            frame = space_manager.draw_spaces(frame)

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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        grabber.release()
        database.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
