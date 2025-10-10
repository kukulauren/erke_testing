from ultralytics import YOLO
import cv2
import math
import time
from dotenv import load_dotenv
import os
load_dotenv()
frame_width = float(os.getenv("FRAME_WIDTH"))
frame_height=float(os.getenv("FRAME_HEIGHT"))
class Yolo:
    def __init__(self, model_path, confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def capture_video(self):
        self.cap = cv2.VideoCapture(0)

    def preprocess(self):
        self.cap.set(3, frame_width)
        self.cap.set(4, frame_height)

    def predict(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        results = self.model(frame, conf=self.confidence)[0]
        return results, frame

    def postprocess_result(self, results, frame):
        classNames = ["cashier", "customer", "scanner", "item", "phone", "cash"]
        output = []
        boxes = results.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            classname = classNames[cls]

            label = f"""
            confidence: {confidence}
            classname: {classname}
            x1: {x1}
            x2: {x2}
            y1: {y1}
            y2: {y2}
            """
            output.append(label)

        return "\n".join(output)

    def test(self, duration=10):
        self.capture_video()
        self.preprocess()
        self.running = True
        start_time = time.time()

        while self.running and (time.time() - start_time < duration):
            results, frame = self.predict()
            if frame is None:
                break
            labels = self.postprocess_result(results, frame)
            print(labels)

    def run(self):
        self.capture_video()
        self.preprocess()
        self.running = True

        while self.running:
            results, frame = self.predict()
            if frame is None:
                break
            labels = self.postprocess_result(results, frame)
            print(labels)

            cv2.imshow("Detected result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()