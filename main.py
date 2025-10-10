from app.pipeline import Yolo
from dotenv import load_dotenv
import os
load_dotenv()
model_path = os.getenv("MODEL_PATH")
confidence = float(os.getenv("CONFIDENCE"))
test_duration=float(os.getenv("TEST_DURATION"))
model=Yolo(model_path)
model.test(test_duration)