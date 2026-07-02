from ultralytics import YOLO

# Load the pretrained YOLO model once when the application starts.
model = YOLO("yolov8n.pt")

def detect(frame):
    """
    Runs object detection on an OpenCV image (frame)
    and returns YOLO results.
    """
    results = model(frame)
    return results