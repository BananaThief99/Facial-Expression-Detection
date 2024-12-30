import os
import cv2
import json
from face_detection import detect_faces_with_deepface
from face_recognition import generate_embedding, match_embedding, update_embeddings
from emotion_detection import predict_emotions
from deepface import DeepFace


# Global configurations
RESULTS_FOLDER = 'static/results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Emotions list
emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']

def process_frame(frame, model):
    """Process a single frame for face detection, recognition, and emotion prediction."""
    boxes, confidences = detect_faces_with_deepface(frame)
    frame_results = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]
        embedding = generate_embedding(face)
        track_id = match_embedding(embedding)

        emotion, confidences = predict_emotions(face, model)
        formatted_confidences = {emotion: round(float(conf) * 100, 2) for emotion, conf in zip(emotions, confidences)}

        frame_results.append({
            'id': track_id,
            'emotion': emotion,
            'confidences': formatted_confidences,
            'bbox': box
        })

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame_results, frame

def process_image(file_path, model):
    """Process an image for face detection, recognition, and emotion prediction."""
    image = cv2.imread(file_path)
    frame_results, annotated_image = process_frame(image, model)

    result_image_path = os.path.join(RESULTS_FOLDER, os.path.basename(file_path))
    result_json_path = os.path.join(RESULTS_FOLDER, os.path.splitext(os.path.basename(file_path))[0] + "_results.json")

    cv2.imwrite(result_image_path, annotated_image)
    with open(result_json_path, 'w') as f:
        json.dump(frame_results, f, indent=4)

    return result_image_path, result_json_path, frame_results

def process_video(file_path, model):
    """Process a video for face detection, recognition, and emotion prediction."""
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    result_video_filename = f'{base_filename}_processed.mp4'
    result_video_path = os.path.join(RESULTS_FOLDER, result_video_filename)
    result_json_path = os.path.join(RESULTS_FOLDER, f'{base_filename}_results.json')

    out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        update_embeddings()  # Clean up expired embeddings

        frame_results, annotated_frame = process_frame(frame, model)
        results.append({'frame': len(results), 'results': frame_results})
        out.write(annotated_frame)

    cap.release()
    out.release()

    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    return result_video_path, result_json_path
