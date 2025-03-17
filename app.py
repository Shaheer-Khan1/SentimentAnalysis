from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace
import os
import numpy as np
import json
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from datetime import datetime

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("skillspark-7a7ee-firebase-adminsdk-fbsvc-efe78bf7a4.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Configure screenshots directory
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
print(f"📁 Screenshots will be saved to: {SCREENSHOTS_DIR}")

# Configure the URL prefix for accessing screenshots
# This should be the base URL where your Flask app is running
BASE_URL = "http://localhost:3000"  # Change this to your actual domain in production

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(data):
    """ Recursively convert NumPy data types to Python native types """
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(v) for v in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return convert_numpy_types(data.tolist())
    elif isinstance(data, (np.bool_)):
        return bool(data)
    return data

def save_screenshot_locally(screenshot_file):
    """Save screenshot to local directory and return the filename and URL"""
    # Generate unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]  # Use shorter UUID for filenames
    filename = f"screenshot_{timestamp}_{unique_id}.jpg"
    
    # Save the file
    file_path = os.path.join(SCREENSHOTS_DIR, filename)
    screenshot_file.save(file_path)
    
    # Generate URL for accessing the file
    screenshot_url = f"{BASE_URL}/screenshots/{filename}"
    
    return filename, screenshot_url

# Route to serve screenshot files
@app.route('/screenshots/<filename>')
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOTS_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict_emotion():
    print(f"📡 Received Request: {request.content_type}")

    if 'image' not in request.files:
        print("🚨 No image found in request!")
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    print(f"✅ Received image: {image_file.filename}")

    # Handle screenshot if provided
    screenshot_filename = None
    screenshot_url = None
    if 'screenshot' in request.files:
        screenshot_file = request.files['screenshot']
        print(f"✅ Received screenshot: {screenshot_file.filename}")
        
        try:
            # Save screenshot locally and get its URL
            screenshot_filename, screenshot_url = save_screenshot_locally(screenshot_file)
            print(f"📸 Screenshot saved as: {screenshot_filename}")
            print(f"🔗 Screenshot URL: {screenshot_url}")
        except Exception as screenshot_error:
            print(f"⚠️ Error saving screenshot: {str(screenshot_error)}")

    # Save the face image temporarily
    image_path = "temp.jpg"
    image_file.save(image_path)

    try:
        # Face detection
        result = DeepFace.analyze(
            image_path,
            actions=['emotion'],
            detector_backend="mtcnn",
            enforce_detection=True
        )

        # Convert entire result object to Python native types before any processing
        result_json = json.dumps(result, cls=NumpyEncoder)
        result_cleaned = json.loads(result_json)
        
        # Store in Firestore
        try:
            doc_data = {
                "image_name": image_file.filename,
                "analysis": result_cleaned[0] if isinstance(result_cleaned, list) else result_cleaned,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            
            # Add screenshot URL if available
            if screenshot_url:
                doc_data["screenshot_url"] = screenshot_url
                doc_data["screenshot_filename"] = screenshot_filename
            
            doc_ref = db.collection("emotion_analysis").document()
            doc_ref.set(doc_data)
            print(f"📤 Saved to Firestore: {doc_ref.id}")
            
            # Include document ID in response
            response_data = json.loads(result_json)
            if isinstance(response_data, list):
                response_data[0]['document_id'] = doc_ref.id
                if screenshot_url:
                    response_data[0]['screenshot_url'] = screenshot_url
            else:
                response_data['document_id'] = doc_ref.id
                if screenshot_url:
                    response_data['screenshot_url'] = screenshot_url
                    
            return app.response_class(
                response=json.dumps(response_data, cls=NumpyEncoder),
                status=200,
                mimetype='application/json'
            )
            
        except Exception as db_error:
            print(f"💾 Database Error: {str(db_error)}")
            # Continue execution even if database save fails
        
        # Return the cleaned result if database operation failed
        return app.response_class(
            response=result_json,
            status=200,
            mimetype='application/json'
        )

    except ValueError as e:
        if "Face could not be detected" in str(e):
            print("⚠️ No face detected in the image.")
            return jsonify({'error': 'No face detected, please provide a clear face image'}), 400
        else:
            print(f"❌ Other Error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Make sure we clean up the temp file even if there's an error
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)