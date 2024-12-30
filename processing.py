import cv2
import os
import json
import numpy as np
from deepface import DeepFace
from utils import predict_emotions
from scipy.spatial.distance import cosine

# Global configurations
RESULTS_FOLDER = 'static/results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']
SIMILARITY_THRESHOLD = 0.5
EMBEDDING_RETENTION_FRAMES = 10  # Set to -1 for indefinite retention

# Global variables for embedding management
stored_embeddings = {}  # {track_id: {"embedding": embedding, "frames_left": retention_frames, "facial_area": ..., "face_confidence": ...}}
track_id_counter = 0  # Start counter for assigning track IDs


def detect_faces(frame):
    """Detect faces using DeepFace's extract_faces method."""
    detections = DeepFace.extract_faces(frame, detector_backend="retinaface", enforce_detection=False)

    boxes, confidences = [], []
    for detection in detections:
        # Ensure 'facial_area' exists and unpack it correctly
        facial_area = detection.get("facial_area", {})
        if not facial_area:
            continue  # Skip detections without a valid facial area

        x1 = facial_area["x"]
        y1 = facial_area["y"]
        x2 = x1 + facial_area["w"]
        y2 = y1 + facial_area["h"]

        # Extract confidence
        confidence = detection.get("face_confidence", 1.0)

        # Append the bounding box and confidence score
        boxes.append([x1, y1, x2, y2])
        confidences.append(confidence)

    return boxes, confidences


def generate_embedding(face):
    """Generate embedding for a given face using DeepFace."""
    embeddings = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
    # Extract the actual embedding vector
    return embeddings[0] if embeddings else None


def match_embedding(current_embedding):
    """
    Match the current embedding with stored embeddings to assign a track ID.
    Args:
        current_embedding (dict): The embedding structure of the current face.
    Returns:
        int: The track ID of the matched embedding or a new ID if no match.
    """
    global stored_embeddings, track_id_counter

    best_similarity = -1
    best_track_id = None

    # Extract the actual embedding vector
    embedding_vector = current_embedding["embedding"] if "embedding" in current_embedding else None
    if not embedding_vector:
        return None

    current_embedding_vector = np.array(embedding_vector)

    for track_id, data in stored_embeddings.items():
        known_embedding_vector = np.array(data["embedding"])

        # Compute cosine similarity
        similarity = 1 - cosine(known_embedding_vector, current_embedding_vector)

        if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
            best_similarity = similarity
            best_track_id = track_id

    if best_track_id is not None:
        # Update the embedding and reset frames_left
        stored_embeddings[best_track_id]["embedding"] = embedding_vector
        stored_embeddings[best_track_id]["frames_left"] = EMBEDDING_RETENTION_FRAMES
        return best_track_id
    else:
        # Assign a new track ID
        track_id_counter += 1
        stored_embeddings[track_id_counter] = {
            "embedding": embedding_vector,
            "frames_left": EMBEDDING_RETENTION_FRAMES,
            "facial_area": current_embedding.get("facial_area", {}),
            "face_confidence": current_embedding.get("face_confidence", 1.0),
        }
        return track_id_counter

def update_embeddings():
    """Update stored embeddings by decreasing retention frames and cleaning expired embeddings."""
    global stored_embeddings
    
    if EMBEDDING_RETENTION_FRAMES == -1:
        return
    
    for track_id in list(stored_embeddings.keys()):
        stored_embeddings[track_id]["frames_left"] -= 1
        if stored_embeddings[track_id]["frames_left"] <= 0:
            del stored_embeddings[track_id]

def process_image(file_path, model):
    """Process an image for face detection and emotion prediction."""
    image = cv2.imread(file_path)
    boxes, confidences = detect_faces(image)
    results = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        face = image[y1:y2, x1:x2]
        embedding = generate_embedding(face)

        if embedding:
            track_id = match_embedding(embedding)

            emotion, confidences = predict_emotions(face, model)
            formatted_confidences = {emotion: round(conf.item() * 100, 2) for emotion, conf in zip(emotions, confidences)}

            results.append({
                'id': track_id,
                'emotion': emotion,
                'confidences': formatted_confidences,
                'bbox': box
            })

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"ID: {track_id} {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    result_image_path = os.path.join(RESULTS_FOLDER, os.path.basename(file_path))
    result_json_path = os.path.join(RESULTS_FOLDER, os.path.splitext(os.path.basename(file_path))[0] + "_results.json")

    cv2.imwrite(result_image_path, image)
    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    return result_image_path, result_json_path, results


def process_video(file_path, model):
    """Process a video for face detection and emotion prediction."""
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

        boxes, confidences = detect_faces(frame)
        frame_results = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]
            embedding = generate_embedding(face)

            if embedding:
                track_id = match_embedding(embedding)

                emotion, confidences = predict_emotions(face, model)
                formatted_confidences = {emotion: round(conf.item() * 100, 2) for emotion, conf in zip(emotions, confidences)}

                frame_results.append({
                    'id': track_id,
                    'emotion': emotion,
                    'confidences': formatted_confidences,
                    'bbox': box
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        results.append({'frame': len(results), 'results': frame_results})
        out.write(frame)

    cap.release()
    out.release()

    with open(result_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    return result_video_path, result_json_path
