from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from utils import load_model, detect_faces, predict_emotions

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_media'
RESULTS_FOLDER = 'static/results'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def get_model_path(model_name):
    # Return the full path of the selected model
    return os.path.join(MODEL_FOLDER, model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'model' not in request.form:
        return redirect(request.url)

    file = request.files['file']
    model_name = request.form['model']
    
    if file.filename == '':
        return redirect(request.url)

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load the selected model
    model_path = get_model_path(model_name)
    model = load_model(model_path)

    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return process_image(file_path, model)
    elif file.filename.lower().endswith(('.mp4', '.avi')):
        return process_video(file_path, model)
    else:
        return "Unsupported file type", 400

def process_image(file_path, model):
    image = cv2.imread(file_path)
    faces, boxes = detect_faces(image)

    emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
    results = []
    for idx, face in enumerate(faces):
        emotion, confidences = predict_emotions(face, model)
        # Find the highest confidence and update the predicted emotion
        max_confidence_idx = torch.argmax(confidences).item()
        predicted_emotion = emotions[max_confidence_idx]
        formatted_confidences = {emotions[i]: round(confidences[i].item() * 100, 2) for i in range(len(emotions))}
        results.append({
            'id': idx + 1,
            'emotion': predicted_emotion,
            'confidences': formatted_confidences
        })
        # Draw bounding box on the image
        x, y, w, h = boxes[idx]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    result_image_path = os.path.join(RESULTS_FOLDER, os.path.basename(file_path))
    cv2.imwrite(result_image_path, image)
    return render_template('results.html', media_type='image', image_path=result_image_path, results=results)


def process_video(file_path, model):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    result_video_path = os.path.join(RESULTS_FOLDER, 'processed_' + os.path.basename(file_path))
    out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    results = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces, boxes = detect_faces(frame)
        frame_results = []
        for idx, face in enumerate(faces):
            emotions, confidences = predict_emotions(face, model)
            frame_results.append({
                'id': idx + 1,
                'emotion': emotions,
                'confidences': confidences.tolist()
            })
            # Draw bounding box
            x, y, w, h = boxes[idx]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        results.append({'frame': frame_idx, 'results': frame_results})
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    result_json_path = os.path.join(RESULTS_FOLDER, 'results.json')
    with open(result_json_path, 'w') as f:
        f.write(jsonify(results).get_data(as_text=True))

    return render_template('results.html', media_type='video', video_path=result_video_path, json_path=result_json_path)

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
