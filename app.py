"""
Deepfake Audio Detector - Flask Backend API
=============================================
Serves the prediction API and static frontend.

Run: python app.py
Then open: http://localhost:5000
"""

import os
import uuid
import time
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from inference import predict_audio, generate_spectrogram

app = Flask(__name__, static_folder="static")

UPLOAD_FOLDER   = "uploads"
SPEC_FOLDER     = "static/spectrograms"
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPEC_FOLDER,   exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Save uploaded file
    uid      = str(uuid.uuid4())[:8]
    filename = secure_filename(f"{uid}_{file.filename}")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        start_time = time.time()

        # Run prediction
        result = predict_audio(filepath)

        # Generate spectrogram
        spec_filename = f"{uid}_spec.png"
        spec_path     = os.path.join(SPEC_FOLDER, spec_filename)
        generate_spectrogram(filepath, spec_path)
        result["spectrogram_url"] = f"/static/spectrograms/{spec_filename}"

        result["processing_time"] = round(time.time() - start_time, 2)

        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({
            "error": str(e),
            "hint": "Train the model first: python model.py"
        }), 503

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/health")
def health():
    model_ready = os.path.exists("models/rf_model.pkl")
    return jsonify({
        "status":      "online",
        "model_ready": model_ready,
        "message":     "Model loaded ✅" if model_ready else "Model not trained yet ⚠️"
    })


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🎙️  Deepfake Audio Detector - Web App")
    print("=" * 50)
    print("  URL: http://localhost:5000")
    print("  API: POST /predict  (multipart audio file)")
    if not os.path.exists("models/rf_model.pkl"):
        print("\n  ⚠️  Warning: Model not trained yet!")
        print("  Run: python model.py   to train first.")
    print("=" * 50 + "\n")
    import os
port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port)