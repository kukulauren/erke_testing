from ultralytics import YOLO
import cv2
import math
import time
from dotenv import load_dotenv
import os
import threading

load_dotenv()
try:
    frame_width = int(float(os.getenv("FRAME_WIDTH", 640)))
except Exception:
    frame_width = 640
try:
    frame_height = int(float(os.getenv("FRAME_HEIGHT", 480)))
except Exception:
    frame_height = 480


class Yolo:
    def __init__(self, model_path, rtsp_path, confidence=0.7):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.rtsp_path = rtsp_path
        self.running = False
        self.cap = None
        self._lock = threading.Lock()
        self.thread = None

    def capture_video(self, reconnect_attempts=3, reconnect_delay=2.0):
        attempt = 0
        while attempt < reconnect_attempts:
            self.cap = cv2.VideoCapture(self.rtsp_path)
            if self.cap.isOpened():
                return True
            attempt += 1
            print(f"Failed to open RTSP stream (attempt {attempt}/{reconnect_attempts}), retrying in {reconnect_delay}s...")
            try:
                self.cap.release()
            except Exception:
                pass
            time.sleep(reconnect_delay)

        # final attempt
        self.cap = cv2.VideoCapture(self.rtsp_path)
        if not self.cap.isOpened():
            print("Failed to open RTSP stream after retries")
            return False
        return True

    def preprocess(self):
        # set frame size if capture is available
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        except Exception:
            pass

    def predict(self):
        if not self.cap:
            return None, None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None
        # run inference in try/except to avoid crashing the loop on unexpected errors
        try:
            results = self.model(frame, conf=self.confidence)[0]
        except Exception as e:
            print(f"Inference error: {e}")
            return None, None
        return results, frame

    def postprocess_result(self, results, frame):
        classNames = ["cashier", "customer", "scanner", "item", "phone", "cash"]
        output = []
        boxes = getattr(results, "boxes", [])
        for box in boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                classname = classNames[cls] if cls < len(classNames) else str(cls)
                label = f"confidence: {confidence}, classname: {classname}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}"
                output.append(label)
            except Exception:
                continue
        return "\n".join(output)

    def start_prediction(self):
        """Main prediction loop. This is safe to run in a background thread.

        It will attempt reconnection if reads fail and will keep running until `stop_prediction` is called.
        """
        with self._lock:
            if self.running:
                print("Prediction already running")
                return
            self.running = True

        if not self.capture_video(reconnect_attempts=2, reconnect_delay=1.0):
            print("Unable to open video source - exiting prediction loop")
            with self._lock:
                self.running = False
            return

        self.preprocess()
        print("Prediction started...")
        while True:
            with self._lock:
                if not self.running:
                    break
            results, frame = self.predict()
            if frame is None:
                print("Frame read failed, attempting to reconnect...")
                try:
                    if self.cap:
                        self.cap.release()
                except Exception:
                    pass
                if not self.capture_video(reconnect_attempts=3, reconnect_delay=2.0):
                    print("Reconnect failed, stopping prediction")
                    break
                self.preprocess()
                # continue to next iteration after reconnect
                continue

            labels = self.postprocess_result(results, frame)
            if labels:
                print(labels)
            time.sleep(0.05)

        self.cleanup()

    def stop_prediction(self):
        print("Stopping prediction...")
        with self._lock:
            self.running = False

    def cleanup(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Prediction stopped and resources released.")
