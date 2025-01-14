from flask import Flask, request, render_template, redirect, Response
import os
from process_media import process_image, process_video
from emotion_detection import load_emotion_model
from deepface import DeepFace
import json
import glob

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploaded_media'
RESULTS_FOLDER = 'static/results'
MODEL_FOLDER = 'models'
DATABASE_FOLDER = 'image_database'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_database', methods=['GET', 'POST'])
def create_database():
    if request.method == 'GET':
        return render_template('create_database.html')
    elif request.method == 'POST':
        db_name = request.form.get('db_name').strip()
        name_list = request.form.get('name_list').strip()

        if not db_name or not name_list:
            return "Database name and name list are required!", 400

        # Ensure the database name is valid for a folder
        safe_db_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in db_name)
        db_folder_path = os.path.join(DATABASE_FOLDER, safe_db_name)

        # Create the main database folder
        os.makedirs(db_folder_path, exist_ok=True)

        # JSON file path inside the database folder
        json_file_path = os.path.join(db_folder_path, f"{safe_db_name}.json")

        # Split the input into a list of names
        names = [name.strip() for name in name_list.splitlines() if name.strip()]

        # Construct the JSON structure
        embeddings = {}
        for i, name in enumerate(names, start=1):
            embeddings[str(i)] = {
                "name": name,
                "images": [],
                "embedding": []
            }

            # Create a subfolder for each name
            person_folder = os.path.join(db_folder_path, name)
            os.makedirs(person_folder, exist_ok=True)

        # Save the JSON file inside the database folder
        with open(json_file_path, 'w') as f:
            json.dump(embeddings, f, indent=4)

        return f"Database '{safe_db_name}' created successfully with {len(names)} entries. Stored in {db_folder_path}."

@app.route('/manage_database', methods=['GET', 'POST'])
def manage_database():
    databases = [f for f in os.listdir(DATABASE_FOLDER) if os.path.isdir(os.path.join(DATABASE_FOLDER, f))]
    database_content = None
    selected_db = None

    if request.method == 'POST':
        selected_db = request.form.get('database_name')
        db_folder_path = os.path.join(DATABASE_FOLDER, selected_db)
        json_file_path = os.path.join(db_folder_path, f"{selected_db}.json")

        if not selected_db or not os.path.exists(db_folder_path) or not os.path.exists(json_file_path):
            return "Invalid database selection!", 400

        # Load the current JSON file
        with open(json_file_path, 'r') as f:
            database_content = json.load(f)

        # Synchronize JSON with the folder contents
        for record_id, record in database_content.items():
            person_folder = os.path.join(db_folder_path, record['name'])
            if os.path.exists(person_folder):
                # Get all image filenames in the person's folder
                image_files = [
                    os.path.basename(image)  # Get only the filename
                    for image in glob.glob(os.path.join(person_folder, '*'))
                    if image.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]

                # Remove nonexistent files from JSON
                record['images'] = [img for img in record['images'] if os.path.exists(os.path.join(person_folder, img))]

                # Add new filenames to the JSON
                for image_file in image_files:
                    if image_file not in record['images']:
                        record['images'].append(image_file)

        # Save the updated JSON file
        with open(json_file_path, 'w') as f:
            json.dump(database_content, f, indent=4)

    return render_template(
        'database_manager.html',
        databases=databases,
        selected_db=selected_db,
        database_content=database_content,
    )


@app.route('/add_images', methods=['POST'])
def add_images():
    # Retrieve data from the request
    person_id = request.form.get('person_id')
    database_name = request.form.get('database_name')
    files = request.files.getlist('images')

    # Validate input
    if not person_id or not database_name or not files:
        return {"success": False, "message": "Invalid request!"}, 400

    # Paths for the database and JSON file
    db_folder_path = os.path.join(DATABASE_FOLDER, database_name)
    json_file_path = os.path.join(db_folder_path, f"{database_name}.json")

    # Load the current database content
    with open(json_file_path, 'r') as f:
        database_content = json.load(f)

    # Ensure the person ID is valid
    if person_id not in database_content:
        return {"success": False, "message": "Invalid person ID!"}, 400

    # Get the folder for the specific person
    person_name = database_content[person_id]['name']
    person_folder = os.path.join(db_folder_path, person_name)

    # Process and save the images
    for file in files:
        file_path = os.path.join(person_folder, file.filename)
        file.save(file_path)  # Save the image file
        if file.filename not in database_content[person_id]['images']:
            database_content[person_id]['images'].append(file.filename)

    # Save the updated JSON content back to the file
    with open(json_file_path, 'w') as f:
        json.dump(database_content, f, indent=4)

    # Return the updated image count and list of images
    return {
        "success": True,
        "image_count": len(database_content[person_id]['images']),
        "images": database_content[person_id]['images']
    }, 200



@app.route('/delete_images', methods=['POST'])
def delete_images():
    person_id = request.form.get('person_id')
    database_name = request.form.get('database_name')
    image_name = request.form.get('image_name')

    if not person_id or not database_name or not image_name:
        return {"success": False, "message": "Invalid request!"}, 400

    db_folder_path = os.path.join(DATABASE_FOLDER, database_name)
    json_file_path = os.path.join(db_folder_path, f"{database_name}.json")

    # Load the database JSON file
    with open(json_file_path, 'r') as f:
        database_content = json.load(f)

    person_name = database_content[person_id]['name']
    person_folder = os.path.join(db_folder_path, person_name)
    image_path = os.path.join(person_folder, image_name)

    # Remove the image file and update JSON
    if os.path.exists(image_path):
        os.remove(image_path)
    if image_name in database_content[person_id]['images']:
        database_content[person_id]['images'].remove(image_name)

    # Save the updated JSON file
    with open(json_file_path, 'w') as f:
        json.dump(database_content, f, indent=4)

    return {
        "success": True,
        "image_count": len(database_content[person_id]['images'])
    }, 200



@app.route('/get_table_data', methods=['GET'])
def get_table_data():
    database_name = request.args.get('database_name')
    if not database_name:
        return {"success": False, "message": "Database name is required!"}, 400

    db_folder_path = os.path.join(DATABASE_FOLDER, database_name)
    json_file_path = os.path.join(db_folder_path, f"{database_name}.json")

    if not os.path.exists(json_file_path):
        return {"success": False, "message": "Database not found!"}, 404

    # Load the database JSON file
    with open(json_file_path, 'r') as f:
        database_content = json.load(f)

    # Prepare the data for the frontend
    table_data = []
    for record_id, record in database_content.items():
        table_data.append({
            "id": record_id,
            "name": record['name'],
            "image_count": len(record['images']),
            "images": record['images']
        })

    return {"success": True, "data": table_data}, 200


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
