import tensorflow as tf
from keras.utils import img_to_array, load_img
import numpy as np
import cv2
import os

# 1) Carga del modelo y del detector de caras
model_file = "C:/Users/kevin/Proyectos/models/emotion_model.h5"
model = tf.keras.models.load_model(model_file)

haarCascadeFile = "C:/Users/kevin/Proyectos/models/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haarCascadeFile)

# 2) Obtener categorías de emociones (en inglés)
trainPath = "C:/Users/kevin/archive/train"
categories = sorted(os.listdir(trainPath))
print("Categories (en inglés):", categories)

# 3) Mapeo de etiquetas al español
label_map = {
    'angry':    'enojado',
    'disgust':  'asco',
    'fear':     'miedo',
    'happy':    'feliz',
    'neutral':  'neutral',
    'sad':      'triste',
    'surprise': 'sorpresa'
}

# 4) Función para preprocesar la región de interés (ROI)
def prepareImageForModel(faceImage):
    # Redimensiona a 48x48 y normaliza
    resized = cv2.resize(faceImage, (48, 48), interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.0
    return imgResult

# 5) Abrir la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Iniciando captura. Pulsa 'q' para salir.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en el fotograma
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extraer ROI de la cara
        roiGray = gray[y:y+h, x:x+w]
        imgForModel = prepareImageForModel(roiGray)

        # Predicción con el modelo
        resultArray = model.predict(imgForModel, verbose=0)
        answer = np.argmax(resultArray, axis=1)[0]
        emotion_en = categories[answer]           # etiqueta en inglés
        emotion_es = label_map[emotion_en]        # traducción al español

        # Dibujar rectángulo y texto en español
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_es, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el fotograma con las anotaciones
    cv2.imshow("Detección de Emoción - Webcam", frame)

    # Salir cuando el usuario presione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6) Liberar recursos
cap.release()
cv2.destroyAllWindows()
