from flask import Flask, request, render_template, redirect, Response
import os
from process_media import process_image, process_video
from emotion_detection import load_emotion_model
from deepface import DeepFace

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploaded_media'
RESULTS_FOLDER = 'static/results'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = load_emotion_model(model_path)

    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process image
        result_image_path, result_json_path, results = process_image(file_path, model)
        print(results)  # For debugging
        return render_template('results.html', media_type='image', 
                               image_path=result_image_path, 
                               json_path=result_json_path, 
                               results=results)
    elif file.filename.lower().endswith(('.mp4', '.avi')):
        # Process video
        result_video_path, result_json_path = process_video(file_path, model)
        return render_template('results.html', media_type='video', 
                               video_path=result_video_path, 
                               json_path=result_json_path)
    else:
        return "Unsupported file type", 400

if __name__ == '__main__':
    app.run(debug=True)
