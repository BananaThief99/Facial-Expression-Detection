import torch
import cv2
import numpy as np
from torchvision import transforms
from networks.DDAM import DDAMNet

def load_model(model_path):
    num_classes = 7
    if 'ferPlus' in model_path:
        num_classes = 8
    model = DDAMNet(num_classes, pretrained=False)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = [image[y:y+h, x:x+w] for (x, y, w, h) in boxes]
    return faces, boxes

def predict_emotions(face, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    face_tensor = transform(face).unsqueeze(0)

    with torch.no_grad():
        output = model(face_tensor)
        if isinstance(output, tuple):
            output = output[0]
        confidences = torch.nn.functional.softmax(output, dim=1)[0]
        emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
        predicted_emotion = emotions[torch.argmax(confidences).item()]
        return predicted_emotion, confidences
