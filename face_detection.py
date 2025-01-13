import cv2
from deepface import DeepFace

def detect_faces_with_deepface(frame):
    """Detect faces using DeepFace's extract_faces method."""
    detections = DeepFace.extract_faces(frame, detector_backend="retinaface", enforce_detection=False)
    boxes, confidences = [], []
    for detection in detections:
        facial_area = detection.get("facial_area", {})
        if not facial_area:
            continue
        x1, y1 = facial_area["x"], facial_area["y"]
        x2, y2 = x1 + facial_area["w"], y1 + facial_area["h"]
        confidence = detection.get("confidence", 1.0)
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)
    return boxes, confidences
