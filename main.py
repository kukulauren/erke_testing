from app.pipeline import Yolo
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import threading
import os

load_dotenv()
model_path = os.getenv("MODEL_PATH")
rtsp_path = os.getenv("RTSP_PATH")
confidence = float(os.getenv("CONFIDENCE"))
model = Yolo(model_path, rtsp_path, confidence)
app = Flask(__name__)

prediction_thread = None


@app.route("/start_prediction", methods=["POST"])
def start_prediction():
    global prediction_thread
    try:
        data = request.get_json(force=True)
        start_timestamps = data.get("start_timestamps")

        if not isinstance(start_timestamps, str):
            return jsonify({"error": "Invalid input. setup timestamps"}), 400

        if model.running:
            return jsonify({"message": "Prediction already running"}), 400

        prediction_thread = threading.Thread(target=model.start_prediction)
        prediction_thread.start()
        return jsonify({"message": "Prediction started"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/stop_prediction", methods=["POST"])
def stop_prediction():
    try:
        data = request.get_json(force=True)
        start_timestamps = data.get("start_timestamps")

        if not isinstance(start_timestamps, str):
            return jsonify({"error": "Invalid input. setup timestamps"}), 400

        model.stop_prediction()
        return jsonify({"message": "Prediction stopped"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
