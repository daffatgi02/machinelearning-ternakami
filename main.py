import os
from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inisialisasi InferenceHTTPClient menggunakan variabel environment
CLIENT = InferenceHTTPClient(
    api_url=os.environ.get("API_URL"),
    api_key=os.environ.get("API_KEY")
)

# Mapping dari hasil prediksi ke deskripsi yang lebih deskriptif
PREDICTION_MAPPING = {
    "pink-eye": "Mata Kambing Terjangkit Pinkeye",
    "normal": "Mata Kambing Terlihat Sehat"
}

@app.route('/')
def home():
    return "Service aktif"

@app.route('/api/predict', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    animal_name = request.form.get('Animal_Name', '')
    type_ = request.form.get('type', 'kambing')

    # Simpan file gambar
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    try:
        # Lakukan inferensi menggunakan InferenceHTTPClient
        result = CLIENT.infer(image_path, model_id="pink-eye-on-goat/1")
        
        # Hapus gambar setelah proses inferensi selesai
        os.remove(image_path)

        # Ekstrak informasi yang dibutuhkan
        predicted_class = result['predicted_classes'][0]
        confidence = result['predictions'][predicted_class]['confidence']

        # Mapping hasil prediksi ke deskripsi yang lebih deskriptif
        descriptive_class = PREDICTION_MAPPING.get(predicted_class, "Unknown")

        response = {
            "Animal_Name": animal_name,
            "predicted_class": descriptive_class,
            "confidence": confidence
        }
        return jsonify(response), 200
    except Exception as e:
        # Hapus gambar jika terjadi kesalahan
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
