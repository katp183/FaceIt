import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ————— Configuración de rutas de frontend —————
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'RestAPI', 'frontend')

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR,
    static_url_path='/static'
)

# ————— Configuración de CORS —————
CORS_OPTIONS = {
    'origins': 'http://localhost:3000',  # origen de tu frontend en Node.js
    'methods': ['GET', 'POST'],          # métodos permitidos
    'allow_headers': ['Content-Type'],   # cabeceras permitidas
    'supports_credentials': False        # ajusta a True si usas cookies
}
# Aplica CORS a todas las rutas bajo el mismo origen
CORS(app, resources={r"/*": CORS_OPTIONS})

# ————— Content Security Policy —————
@app.after_request
def add_csp(response):
    csp = (
        "default-src 'self'; "
        "script-src 'self'; "
        # Permitimos CSS inline y de Google Fonts
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        # Permitimos cargar las fuentes desde Google
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        # Conexiones XHR/Fetch a ambos backends
        "connect-src 'self' http://localhost:3000 http://localhost:5000"
    )
    response.headers['Content-Security-Policy'] = csp
    return response

# ————— Modelos y cascada Haar —————
MODEL_PATH   = os.path.join(PROJECT_ROOT, 'models', 'emotion_model.h5')
CASCADE_PATH = os.path.join(PROJECT_ROOT, 'models', 'haarcascade_frontalface_default.xml')

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ————— Etiquetas y traducción —————
categories = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_map = {
    'angry':    'enojado',
    'disgust':  'asco',
    'fear':     'miedo',
    'happy':    'feliz',
    'neutral':  'neutral',
    'sad':      'triste',
    'surprise': 'sorpresa'
}

def prepare_image_for_model(face_img):
    resized = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_AREA)
    resized = resized.astype('float32') / 255.0
    img = np.expand_dims(resized, axis=-1)  # (48,48,1)
    img = np.expand_dims(img, axis=0)       # (1,48,48,1)
    return img

def decode_and_detect(data_url):
    header, encoded = data_url.split(',', 1)
    img_data = base64.b64decode(encoded)
    nparr = np.frombuffer(img_data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if not len(faces):
        return None
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    return prepare_image_for_model(roi)

@app.route('/')
def index():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_url = request.json.get('image')
    img_input = decode_and_detect(data_url)
    if img_input is None:
        return jsonify({'error': 'No se detectó cara'}), 400

    preds = model.predict(img_input, verbose=0)[0]
    idx = np.argmax(preds)
    emo_en = categories[idx]
    emo_es = label_map[emo_en]
    return jsonify({'emotion': emo_es})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
