import cv2
import numpy as np
import time
from src.space_manager import SpaceManager

SPACES_FILE = 'config/parking_spaces.json'
WIDTH, HEIGHT = 1280, 720
FPS = 30

def draw_vehicle(frame, center, track_id, label):
    cx, cy = center
    w, h = 80, 50
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 150, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def lerp(a, b, t):
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t))


def get_space_center(sm, name):
    return sm.get_space_center(name)


def main():
    sm = SpaceManager(SPACES_FILE)

    if not sm.spaces:
        print("No spaces loaded. Check config/parking_spaces.json.")
        return

    space_names = list(sm.spaces.keys())

    # Each scenario: list of (track_id, target_space) pairs that arrive over time
    scenarios = [
        [('Car 1', space_names[0])],
        [('Car 1', space_names[0]), ('Car 2', space_names[2])],
        [('Car 1', space_names[0]), ('Car 2', space_names[2]), ('Car 3', space_names[4])],
        [('Car 1', space_names[0]), ('Car 2', space_names[2]), ('Car 3', space_names[4]),
         ('Car 4', space_names[1])],
        [],  # all leave
    ]

    print("Demo running — press Q to quit, SPACE to skip to next scenario.")

    scenario_idx = 0
    scenario_duration = 3.0  # seconds per scenario
    anim_duration = 1.2       # seconds for car to drive in

    t_start = time.time()
    active_vehicles = []      # list of {label, center, target, t_start, done}

    while True:
        elapsed = time.time() - t_start

        # Advance scenario
        if elapsed > scenario_duration:
            scenario_idx = (scenario_idx + 1) % len(scenarios)
            t_start = time.time()
            elapsed = 0.0

            # Build new active vehicles list
            scenario = scenarios[scenario_idx]
            active_vehicles = []
            for i, (label, space) in enumerate(scenario):
                target = get_space_center(sm, space)
                if target:
                    origin = (target[0], -60)  # drop in from above the space
                    active_vehicles.append({
                        'label': label,
                        'space': space,
                        'origin': origin,
                        'target': target,
                        'anim_start': i * 0.3,
                    })

        frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)

        # Compute current positions
        current_vehicles = []
        for v in active_vehicles:
            anim_elapsed = elapsed - v['anim_start']
            t = min(1.0, max(0.0, anim_elapsed / anim_duration))
            pos = lerp(v['origin'], v['target'], t)
            current_vehicles.append({
                'track_id': hash(v['label']) % 1000,
                'center': pos,
                'bbox': [pos[0]-40, pos[1]-25, pos[0]+40, pos[1]+25],
                'class_name': 'car',
                'label': v['label'],
            })

        sm.update_occupancy(current_vehicles)
        frame = sm.draw_spaces(frame)

        for v in current_vehicles:
            draw_vehicle(frame, v['center'], v['track_id'], v['label'])

        summary = sm.get_occupancy_summary()
        cv2.putText(frame, f"Spaces: {summary['occupied']}/{summary['total']} occupied",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        scenario_label = f"Scenario {scenario_idx + 1}/{len(scenarios)}"
        cv2.putText(frame, scenario_label, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow('Parking Demo', frame)
        key = cv2.waitKey(1000 // FPS) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            scenario_idx = (scenario_idx + 1) % len(scenarios)
            t_start = time.time()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
