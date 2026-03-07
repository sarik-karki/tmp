
import time

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class VehicleTracker:

    def __init__(self, model_path='yolov8s.pt', tracker_config='config/bytetrack.yaml',
                 confidence=0.5, fps=30, process_every_n=1, imgsz=416,
                 detector=None):
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.fps = fps
        self.imgsz = imgsz

        # If an external detector (e.g. Hailo) is provided, use it instead of YOLO
        self.detector = detector
        if detector is None:
            if YOLO is None:
                raise ImportError(
                    "ultralytics is not installed and no external detector provided. "
                    "Install with: pip install ultralytics, or use backend: hailo"
                )
            self.model = YOLO(model_path)
        else:
            self.model = None

        self.vehicle_classes = [0, 1, 2, 3]

        self.active_tracks = {}
        self.track_history = {}
        self.exited_tracks = []
        self.entered_tracks = []

        # Config for exit detection
        self.max_missing_frames = 30
        self.max_history_age = 300

        # Frame skipping: only run detection every N frames
        self.process_every_n = max(1, process_every_n)
        self._frame_count = 0
        self._last_vehicles = []

        # Simple track ID assignment for external detector mode
        self._next_track_id = 1
        self._iou_threshold = 0.3

    def update(self, frame):

        self.entered_tracks = []
        self.exited_tracks = []

        self._frame_count += 1

        # On skipped frames, reuse last results but still check exits
        if self.process_every_n > 1 and self._frame_count % self.process_every_n != 0:
            self._check_exits(set(self.active_tracks.keys()))
            return self._last_vehicles

        if self.detector is not None:
            return self._update_with_external_detector(frame)

        results = self.model.track(
            source=frame,
            tracker=self.tracker_config,
            persist=True,
            conf=self.confidence,
            classes=self.vehicle_classes,
            imgsz=self.imgsz,
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

        self._last_vehicles = current_vehicles
        return current_vehicles

    def _update_with_external_detector(self, frame):
        """Run detection via external detector (Hailo) and do simple IoU tracking."""
        detections = self.detector.detect(frame)

        current_vehicles = []
        current_ids = set()
        now = time.time()

        # Match detections to existing tracks by IoU
        unmatched_dets = list(range(len(detections)))
        matched = {}

        if self.active_tracks and detections:
            for track_id, prev in self.active_tracks.items():
                best_iou = 0
                best_idx = -1
                for di in unmatched_dets:
                    iou = self._compute_iou(prev['bbox'], list(detections[di].bbox))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = di
                if best_iou >= self._iou_threshold and best_idx >= 0:
                    matched[track_id] = best_idx
                    unmatched_dets.remove(best_idx)

        # Update matched tracks
        for track_id, di in matched.items():
            det = detections[di]
            bbox = list(det.bbox)
            center = self._get_center(bbox)
            vehicle = {
                'track_id': track_id,
                'bbox': bbox,
                'center': center,
                'confidence': det.conf,
                'class_id': det.cls,
                'class_name': self._get_class_name(det.cls),
                'timestamp': now
            }
            current_vehicles.append(vehicle)
            current_ids.add(track_id)
            self.active_tracks[track_id] = vehicle

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            det = detections[di]
            if self.vehicle_classes and det.cls not in self.vehicle_classes:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            bbox = list(det.bbox)
            center = self._get_center(bbox)
            vehicle = {
                'track_id': track_id,
                'bbox': bbox,
                'center': center,
                'confidence': det.conf,
                'class_id': det.cls,
                'class_name': self._get_class_name(det.cls),
                'timestamp': now
            }
            current_vehicles.append(vehicle)
            current_ids.add(track_id)
            self.active_tracks[track_id] = vehicle
            self.entered_tracks.append(vehicle)
            self.track_history[track_id] = {
                'first_seen': now,
                'entry_position': center,
                'class_name': vehicle['class_name']
            }

        self._check_exits(current_ids)

        self._last_vehicles = current_vehicles
        return current_vehicles

    @staticmethod
    def _compute_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

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
        self._frame_count = 0
        self._last_vehicles = []
        self._next_track_id = 1
        if self.model is not None and YOLO is not None:
            self.model = YOLO(self.model_path)
