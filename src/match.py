import time
import threading
import cv2
import numpy as np
from collections import deque


class PlateMatcher:

    def __init__(self, config):
        self._lock = threading.Lock()
        points = config.get('entry_zone', {}).get('polygon', [])
        self.entry_zone = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        self.plate_queue = deque()
        self.track_plate_map = {}
        self.unmatched_tracks = {}
        self.match_timeout = config.get('plate_reader', {}).get('poll_interval', 1) * 10

        # Precompute bounding rect for fast rejection before polygon test
        if len(self.entry_zone) > 0:
            x, y, w, h = cv2.boundingRect(self.entry_zone)
            self._zone_bbox = (x, y, x + w, y + h)
        else:
            self._zone_bbox = None

    def push_plate(self, plate_text, timestamp=None):
        if not plate_text:
            return
        ts = timestamp or time.time()
        with self._lock:
            self.plate_queue.append({'plate': plate_text.upper().strip(), 'timestamp': ts})
            self._try_match_waiting()

    def try_assign(self, track_id, center):
        with self._lock:
            if track_id in self.track_plate_map:
                return
            if not self._in_entry_zone(center):
                return
            if self.plate_queue:
                entry = self.plate_queue.popleft()
                self.track_plate_map[track_id] = entry['plate']
                print(f"Plate matched: track #{track_id} -> {entry['plate']}")
            else:
                self.unmatched_tracks[track_id] = time.time()

    def get_plate(self, track_id):
        with self._lock:
            return self.track_plate_map.get(track_id)

    def release(self, track_id):
        with self._lock:
            self.track_plate_map.pop(track_id, None)
            self.unmatched_tracks.pop(track_id, None)

    def _in_entry_zone(self, center):
        if self._zone_bbox is None:
            return False
        px, py = float(center[0]), float(center[1])
        x1, y1, x2, y2 = self._zone_bbox
        if px < x1 or px > x2 or py < y1 or py > y2:
            return False
        result = cv2.pointPolygonTest(
            self.entry_zone, (px, py), False
        )
        return result >= 0

    def _try_match_waiting(self):
        if not self.plate_queue or not self.unmatched_tracks:
            return
        now = time.time()
        timed_out = [
            tid for tid, ts in self.unmatched_tracks.items()
            if now - ts > self.match_timeout
        ]
        for tid in timed_out:
            del self.unmatched_tracks[tid]
        if not self.unmatched_tracks:
            return
        oldest_track = min(self.unmatched_tracks, key=self.unmatched_tracks.get)
        entry = self.plate_queue.popleft()
        self.track_plate_map[oldest_track] = entry['plate']
        del self.unmatched_tracks[oldest_track]
        print(f"Late plate match: track #{oldest_track} -> {entry['plate']}")

    def get_all(self):
        return dict(self.track_plate_map)

    def queue_size(self):
        return len(self.plate_queue)
