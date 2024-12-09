from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify, Response, send_from_directory, make_response
import os
from mtcnn import MTCNN
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from torchvision import transforms
from utils import load_model, detect_faces, predict_emotions

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_media'
RESULTS_FOLDER = 'static/results'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

# Initialize the FaceNet model for embedding extraction
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize the DeepSORT tracker
deep_sort_tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

# Store embeddings for active tracks
active_embeddings = {}

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

def process_image(file_path, model):
    image = cv2.imread(file_path)
    faces, boxes = detect_faces(image)
    
    results = []
    for idx, face in enumerate(faces):
        emotion, confidences = predict_emotions(face, model)
        # Find the highest confidence and update the predicted emotion
        max_confidence_idx = torch.argmax(confidences).item()
        predicted_emotion = emotions[max_confidence_idx]
        formatted_confidences = {emotions[i]: round(confidences[i].item() * 100, 2) for i in range(len(emotions))}

        # Add results to the list
        results.append({
            'id': idx + 1,  # Row ID starts from 1
            'emotion': predicted_emotion,
            'confidences': formatted_confidences
        })

        # Draw bounding box and ID label on the image
        x, y, w, h = boxes[idx]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put the ID label beside the bounding box
        label_position = (x, y - 10 if y - 10 > 10 else y + 10)
        cv2.putText(image, f'ID: {idx + 1}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save the result image in the static/results folder
    result_image_path = os.path.join(RESULTS_FOLDER, os.path.basename(file_path))
    cv2.imwrite(result_image_path, image)
    return render_template('results.html', media_type='image', image_path=result_image_path, results=results)

# Initialize the MTCNN detector
mtcnn_detector = MTCNN()

def detect_faces_with_mtcnn(frame):
    """Detect faces using MTCNN."""
    detections = mtcnn_detector.detect_faces(frame)
    boxes = []
    for detection in detections:
        x, y, width, height = detection['box']
        # Ensure box dimensions are within frame boundaries
        x, y = max(0, x), max(0, y)
        boxes.append([x, y, width, height])
    return boxes

def get_face_embedding(face):
    """Extract embedding for a face using FaceNet."""
    face_resized = cv2.resize(face, (160, 160))
    face_tensor = torch.tensor(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        embedding = facenet(face_tensor).squeeze().numpy()
    return embedding

def process_video(file_path, model):
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    result_video_filename = f'{base_filename}_processed.mp4'
    result_video_path = os.path.join("static/results", result_video_filename)
    out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    result_json_path = os.path.join("static/results", f'{base_filename}_results.json')

    results = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using MTCNN
        boxes = detect_faces_with_mtcnn(frame)
        detections = []
        embeddings = []
        for box in boxes:
            x, y, width, height = box
            detections.append([x, y, x + width, y + height])  # Convert to [x1, y1, x2, y2]
            face = frame[y:y+height, x:x+width]
            embeddings.append(get_face_embedding(face))

        # Update tracker with detections
        tracks = deep_sort_tracker.update_tracks(detections, frame=frame)

        frame_results = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Get track ID and bounding box
            track_id = track.track_id
            bbox = track.to_tlbr()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            face = frame[y1:y2, x1:x2]

            # Update or create embedding for the track
            if track_id not in active_embeddings:
                active_embeddings[track_id] = get_face_embedding(face)
            else:
                current_embedding = get_face_embedding(face)
                similarity = 1 - cosine(active_embeddings[track_id], current_embedding)
                if similarity < 0.5:  # Threshold for new identity
                    active_embeddings[track_id] = current_embedding

            # Predict emotion
            emotion, confidences = predict_emotions(face, model)
            formatted_confidences = {emotions[i]: round(confidences[i].item() * 100, 2) for i in range(len(emotions))}

            # Add result
            frame_results.append({
                'id': track_id,
                'emotion': emotion,
                'confidences': formatted_confidences
            })

            # Draw bounding box and emotion label with ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_position = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
            cv2.putText(frame, f"ID: {track_id} {emotion}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        results.append({'frame': frame_idx, 'results': frame_results})
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Write results to JSON file
    with open(result_json_path, 'w') as f:
        f.write(jsonify(results).get_data(as_text=True))

    return render_template('results.html', media_type='video', video_path=result_video_filename, json_path=result_json_path)

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)