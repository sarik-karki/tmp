
from ultralytics import YOLO
import time


class VehicleTracker:

    def __init__(self, model_path='yolov8s.pt', tracker_config='config/bytetrack.yaml', confidence=0.5, fps=30):
        self.model_path = model_path  
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.fps = fps

        self.vehicle_classes = [0, 1, 2, 3]

        self.active_tracks = {}     
        self.track_history = {}      
        self.exited_tracks = []      
        self.entered_tracks = []    

        # Config for exit detection
        self.max_missing_frames = 30  
        self.max_history_age = 300    

    def update(self, frame):
       
        self.entered_tracks = []
        self.exited_tracks = []

        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            persist=True,
            conf=self.confidence,
            classes=self.vehicle_classes,
            verbose=False
        )

        current_vehicles = []
        current_ids = set()

        # Parse results
        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    track_id = int(track_ids[i])
                    current_ids.add(track_id)

                    bbox = boxes[i].tolist()
                    center = self._get_center(bbox)
                    now = time.time()

                    vehicle = {
                        'track_id': track_id,
                        'bbox': bbox,
                        'center': center,
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self._get_class_name(int(class_ids[i])),
                        'timestamp': now
                    }

                    current_vehicles.append(vehicle)

                    if track_id not in self.active_tracks:
                        self.entered_tracks.append(vehicle)
                        self.track_history[track_id] = {
                            'first_seen': now,
                            'entry_position': center,
                            'class_name': vehicle['class_name']
                        }

                    self.active_tracks[track_id] = vehicle

        self._check_exits(current_ids)

        return current_vehicles

    def _check_exits(self, current_ids):

        now = time.time()
        exited_ids = []

        for track_id, vehicle in list(self.active_tracks.items()):
            if track_id not in current_ids:
                seconds_missing = now - vehicle['timestamp']

                if seconds_missing > (self.max_missing_frames / self.fps):
                    exit_info = {
                        'track_id': track_id,
                        'last_position': vehicle['center'],
                        'last_bbox': vehicle['bbox'],
                        'class_name': vehicle['class_name'],
                        'exit_time': now,
                        'duration': now - self.track_history[track_id]['first_seen']
                    }

                    self.exited_tracks.append(exit_info)
                    exited_ids.append(track_id)

                    self.track_history[track_id]['exit_time'] = now
                    self.track_history[track_id]['exit_position'] = vehicle['center']

        for track_id in exited_ids:
            del self.active_tracks[track_id]

        self._prune_history(now)

    def _prune_history(self, now):
        stale_ids = [
            tid for tid, info in self.track_history.items()
            if 'exit_time' in info and (now - info['exit_time']) > self.max_history_age
        ]
        for tid in stale_ids:
            del self.track_history[tid]

    def get_entered(self):
        return self.entered_tracks

    def get_exited(self):
        return self.exited_tracks

    def get_active_count(self):
        return len(self.active_tracks)

    def get_track_info(self, track_id):
        return self.track_history.get(track_id)

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def _get_class_name(self, class_id):
        names = {0: 'bus', 1: 'car', 2: 'motorcycle', 3: 'shuttle'}
        return names.get(class_id, 'vehicle')

    def reset(self):
        self.active_tracks = {}
        self.track_history = {}
        self.exited_tracks = []
        self.entered_tracks = []
        self.model = YOLO(self.model_path)
