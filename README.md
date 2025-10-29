# Table of Contents

1. [Project Description](#project-description)
2. [Installation and Setup](#installation-and-setup)
3. [API Guideline](#api-guideline)

---

# 1. Project Description

## Stream YOLOv11 Pipeline

A real-time object detection pipeline with RTSP using a custom model trained on YOLOv11 (`11l` model), implemented with OpenCV and Ultralytics.

---

### 🔧 Requirements

- `ultralytics`
- `opencv-python` (cv2)
- `math`
- `time`
- `Flask`
- `python-dotenv`

### OOP Pipeline Methods

- `__init__(model_path=None)`: Loads custom model
- `capture_video`: Stream RTSP
- `preprocess`: Width is 640 and height is 480
- `postprocess_result`: Returns class names, confidence, and coordinate points
- `test`: Test the pipeline for 10 seconds (default value, can be set manually)
- `run`: Model will run continuously

### Tested Model File Size

- `best (2).pt` — 51.2MB

### Model Result Example

```
0: 480x640 1 customer, 1 phone, 419.0ms
Speed: 0.7ms preprocess, 419.0ms inference, 0.8ms postprocess per image at shape (1, 3, 480, 640)

    confidence: 0.96
    classname: customer
    x1: 44
    x2: 601
    y1: 6
    y2: 479

    confidence: 0.82
    classname: phone
    x1: 299
    x2: 451
    y1: 0
    y2: 305
```

---

# 2. Installation and Setup

## Running the Project

```bash
git clone <github-link> # or download this zip
cd erke
```

### Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare your `.env` file with required configuration
Add RTSP path in the .env file.

### Run the Pipeline

```bash
python main.py
```

---

# 3. API Guideline

There are two POST methods: `start_prediction` and `stop_prediction`.

## API Endpoints

### 1. `start_prediction`

- **Base URL:** `http://127.0.0.1:8000/`
- **Endpoint:** `/start_prediction`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Key:** `timestamps`
- **Response:**
  - Prediction started

### 2. `stop_prediction`

- **Base URL:** `http://127.0.0.1:8000/`
- **Endpoint:** `/stop_prediction`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Key:** `timestamps`
- **Response:**
  - Prediction stopped

_Note: The logic to sum up the prediction and get an abstracted answer is still under consideration._

CURL setup
Command Prompt
curl -X POST http://127.0.0.1:8000/start_prediction -H "Content-Type: application/json" -d "{\"start_timestamps\": \"2025-10-28T22:00:00\"}"
curl -X POST http://127.0.0.1:8000/stop_prediction -H "Content-Type: application/json" -d "{\"start_timestamps\": \"2025-10-28T22:00:00\"}"

Window Powershell
curl -X POST http://127.0.0.1:8000/start_prediction `
     -H "Content-Type: application/json" `
     -d '{"start_timestamps": "2025-10-28T22:00:00"}'
curl -X POST http://127.0.0.1:8000/start_prediction `
     -H "Content-Type: application/json" `
     -d '{"start_timestamps": "2025-10-28T22:00:00"}'
     
> [!NOTE]
> The timestamps key can be any string or text value.
