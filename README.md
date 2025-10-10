# Stream YOLOv11 Pipeline

A real-time object detection pipeline using custom model trained on YOLOv11 (`11l` model) implemented with OpenCV and Ultralytics.

---

### 🔧 Requirements

- `ultralytics`
- `opencv-python` (cv2)
- `math`
- `time`
- `python-dotenv`

### OOP pipelines will include the following method
- `__init__(model_path=None)`
	-	Loads custom model/
- `capture_video`
  - start the camera
- `preprocess`
  - width is 640 and height is 480.
- `postprocess_result`
  - returns class names, confidence and coordinate points.
- `test`
  - test the pipeline for 10 seconds (default value) which can be passed manually.
- `run`
  - model will run continuously.

### Tested model files' sizes
- best (2).pt 51.2MB

### Model result example

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

# Running the project
```bash
git clone github-link / Download this zip
cd erke
```
### Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### Install dependencies:
```bash
pip install requirements.txt
```

### Prepare your .env file with required configuration
### Run the pipeline
```bash
python main.py
```
> [!WARNING]
> Currently model.test() method is implemented for testing purpose. 
