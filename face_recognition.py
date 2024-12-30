# face_recognition.py
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

SIMILARITY_THRESHOLD = 0.5
EMBEDDING_RETENTION_FRAMES = 10
stored_embeddings = {}
track_id_counter = 0

def generate_embedding(face):
    """Generate embedding for a given face using DeepFace."""
    embedding = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
    return embedding

def match_embedding(current_embedding):
    """
    Match the current embedding with stored embeddings to assign a track ID.
    """
    global stored_embeddings, track_id_counter
    best_similarity = -1
    best_track_id = None

    if isinstance(current_embedding, list) and len(current_embedding) > 0:
        current_embedding = current_embedding[0].get("embedding")

    if not current_embedding:
        raise ValueError("Invalid embedding structure passed to match_embedding.")

    current_embedding = np.array(current_embedding)

    for track_id, data in stored_embeddings.items():
        known_embedding = data["embedding"]
        if isinstance(known_embedding, list) and len(known_embedding) > 0:
            known_embedding = known_embedding[0].get("embedding")

        if not known_embedding:
            continue

        known_embedding = np.array(known_embedding)
        similarity = 1 - cosine(known_embedding, current_embedding)

        if similarity > best_similarity and similarity >= SIMILARITY_THRESHOLD:
            best_similarity = similarity
            best_track_id = track_id

    if best_track_id is not None:
        stored_embeddings[best_track_id]["embedding"] = [{"embedding": current_embedding.tolist()}]
        stored_embeddings[best_track_id]["frames_left"] = EMBEDDING_RETENTION_FRAMES
        return best_track_id
    else:
        track_id_counter += 1
        stored_embeddings[track_id_counter] = {
            "embedding": [{"embedding": current_embedding.tolist()}],
            "frames_left": EMBEDDING_RETENTION_FRAMES
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
