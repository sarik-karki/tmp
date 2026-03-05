import cv2
import numpy as np
import json
from pathlib import Path


class SpaceManager:

    def __init__(self, spaces_file=None):
        self.spaces = {}
        self.space_status = {}
        self.space_occupants = {}

        if spaces_file and Path(spaces_file).exists():
            self.load_spaces(spaces_file)

    def load_spaces(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.spaces = {}
        for name, points in data.items():
            self.spaces[name] = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            self.space_status[name] = 'empty'
            self.space_occupants[name] = None

        print(f"Loaded {len(self.spaces)} parking spaces")

    def save_spaces(self, filepath):
        data = {}
        for name, polygon in self.spaces.items():
            data[name] = polygon.tolist()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.spaces)} parking spaces to {filepath}")

    def add_space(self, name, points):
        self.spaces[name] = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        self.space_status[name] = 'empty'
        self.space_occupants[name] = None

    def remove_space(self, name):
        if name in self.spaces:
            del self.spaces[name]
            del self.space_status[name]
            del self.space_occupants[name]

    def get_space(self, point):
        for name, polygon in self.spaces.items():
            result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
            if result >= 0:
                return name

        return None

    def update_occupancy(self, tracked_vehicles):
        for name in self.spaces:
            self.space_status[name] = 'empty'
            self.space_occupants[name] = None

        for vehicle in tracked_vehicles:
            center = vehicle['center']
            track_id = vehicle['track_id']

            space = self.get_space(center)

            if space:
                if self.space_occupants[space] is not None:
                    print(f"Space {space} already occupied by track {self.space_occupants[space]}, overwriting with track {track_id}")
                self.space_status[space] = 'occupied'
                self.space_occupants[space] = track_id

        result = {}
        for name in self.spaces:
            result[name] = {
                'status': self.space_status[name],
                'track_id': self.space_occupants[name]
            }

        return result

    def get_vehicle_space(self, track_id):
        for name, occupant in self.space_occupants.items():
            if occupant == track_id:
                return name
        return None

    def get_empty_spaces(self):
        return [name for name, status in self.space_status.items() if status == 'empty']

    def get_occupied_spaces(self):
        return [name for name, status in self.space_status.items() if status == 'occupied']

    def get_occupancy_summary(self):
        total = len(self.spaces)
        occupied = len(self.get_occupied_spaces())
        empty = total - occupied

        return {
            'total': total,
            'occupied': occupied,
            'empty': empty,
            'percent_full': (occupied / total * 100) if total > 0 else 0
        }

    def get_space_center(self, space_name):
        if space_name not in self.spaces:
            return None

        polygon = self.spaces[space_name]
        M = cv2.moments(polygon)

        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return (cx, cy)

    def draw_spaces(self, frame, show_labels=True, show_status=True):
        frame = frame.copy()

        colors = {
            'empty': (0, 255, 0),
            'occupied': (0, 0, 255),
            'unknown': (128, 128, 128)
        }

        for name, polygon in self.spaces.items():
            if show_status:
                status = self.space_status.get(name, 'unknown')
                color = colors.get(status, colors['unknown'])
            else:
                color = (255, 255, 0)

            cv2.polylines(frame, [polygon], True, color, 2)

            if show_labels:
                center = self.get_space_center(name)
                if center:
                    label = name
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame,
                                  (center[0] - w//2 - 2, center[1] - h//2 - 2),
                                  (center[0] + w//2 + 2, center[1] + h//2 + 2),
                                  color, -1)
                    cv2.putText(frame, label,
                                (center[0] - w//2, center[1] + h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame
