from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify, Response
import os
import cv2
import numpy as np
import json
from retinaface import RetinaFace
from utils import load_model, predict_emotions

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_media'
RESULTS_FOLDER = 'static/results'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

def get_model_path(model_name):
    # Return the full path of the selected model
    return os.path.join(MODEL_FOLDER, model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<path:filename>')
def video(filename):
    # Path to the video file
    video_path = os.path.join('static', 'results', filename)
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        return Response(status=404)

    # Handle the 'Range' header from the request
    range_header = request.headers.get('Range', None)
    if not range_header:
        # No range request; send the entire file
        with open(video_path, 'rb') as f:
            data = f.read()
        headers = {
            'Content-Type': 'video/mp4',
            'Content-Length': str(os.path.getsize(video_path)),
            'Accept-Ranges': 'bytes'
        }
        return Response(data, headers=headers)

    # Parse the range header (e.g., "bytes=0-1024")
    byte_range = range_header.split("=")[-1]
    start, end = byte_range.split("-")

    # Convert start and end to integers, and set default end if it's empty
    start = int(start)
    file_size = os.path.getsize(video_path)
    end = int(end) if end else file_size - 1

    # Read the specific byte range from the file
    with open(video_path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start + 1)

    # Create response headers for partial content
    headers = {
        'Content-Range': f'bytes {start}-{end}/{file_size}',
        'Accept-Ranges': 'bytes',
        'Content-Length': str(end - start + 1),
        'Content-Type': 'video/mp4',
    }

    # Return the partial content response
    return Response(data, status=206, headers=headers)

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

def detect_faces_with_retinaface(frame):
    """Detect faces using RetinaFace."""
    detections = RetinaFace.detect_faces(frame)
    boxes, confidences = [], []
    for key in detections:
        det = detections[key]
        x1, y1, x2, y2 = map(int, det["facial_area"])
        confidence = det.get("score", 1.0)
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)
    return boxes, confidences

def process_image(file_path, model):
    image = cv2.imread(file_path)
    boxes, confidences = detect_faces_with_retinaface(image)
    results = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        face = image[y1:y2, x1:x2]
        emotion, confidences = predict_emotions(face, model)
        formatted_confidences = {emotions[i]: round(confidences[i].item() * 100, 2) for i in range(len(emotions))}

        results.append({
            'id': idx + 1,
            'emotion': emotion,
            'confidences': formatted_confidences
        })

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"ID: {idx + 1} {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    result_image_path = os.path.join(RESULTS_FOLDER, os.path.basename(file_path))
    cv2.imwrite(result_image_path, image)

    result_json_path = os.path.join(RESULTS_FOLDER, os.path.splitext(os.path.basename(file_path))[0] + "_results.json")
    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    return render_template('results.html', media_type='image', image_path=result_image_path, results=results, json_path=result_json_path)

def process_video(file_path, model):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    result_video_filename = f'{base_filename}_processed.mp4'
    result_video_path = os.path.join("static/results", result_video_filename)
    result_json_path = os.path.join("static/results", f'{base_filename}_results.json')
    out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    results = []  # Store frame-wise results
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using RetinaFace
        boxes, confidences = detect_faces_with_retinaface(frame)

        frame_results = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]
            emotion, confidences = predict_emotions(face, model)
            formatted_confidences = {emotions[i]: round(confidences[i].item() * 100, 2) for i in range(len(emotions))}

            frame_results.append({
                'id': idx + 1,
                'emotion': emotion,
                'confidences': formatted_confidences
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {idx + 1} {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        results.append({'frame': frame_idx, 'results': frame_results})
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    return render_template('results.html', media_type='video', video_path=result_video_filename, json_path=result_json_path)

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
