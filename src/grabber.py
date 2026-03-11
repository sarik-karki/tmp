import threading
import time
import cv2


class LatestFrameGrabber:

    def __init__(self, source=0, backend=None, width=None, height=None,
                 warmup_frames=0, target_fps=None):
        # Convert numeric strings to int for local camera indices
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if backend is not None:
            self.cap = cv2.VideoCapture(source, backend)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        if backend is not None or width is not None or height is not None:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        for _ in range(warmup_frames):
            self.cap.read()
            time.sleep(0.01)

        self.lock = threading.Lock()
        self.frame = None
        self.ok = True
        self.stopped = False
        self._new_frame = False

        # Cap grab rate to avoid burning CPU
        self._frame_interval = 1.0 / target_fps if target_fps else None

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frame
                    self._new_frame = True
            if not ok:
                time.sleep(0.05)
            elif self._frame_interval:
                time.sleep(self._frame_interval)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            self._new_frame = False
            return True, self.frame

    def has_new_frame(self):
        with self.lock:
            return self._new_frame

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()
