import threading
import cv2


class LatestFrameGrabber:

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.lock = threading.Lock()
        self.frame = None
        self.ok = True
        self.stopped = False

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok:
                    self.frame = frame
            if not ok:
                import time
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()
