import tensorflow as tf
from keras.utils import img_to_array, load_img
import numpy as np
import cv2
import os

# 1) Carga del modelo
model_file = "C:/Users/kevin/Proyectos/models/emotion_model.h5"
model = tf.keras.models.load_model(model_file)
print(model.summary())

batchSize = 32

# 2) Obtener categorías de emociones (en inglés)
print("Categories (en inglés):")
trainPath = "C:/Users/kevin/archive/train"
categories = os.listdir(trainPath)
categories.sort()
print(categories)
numOfClasses = len(categories)
print("Número de clases:", numOfClasses)

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

# 4) Función para encontrar la cara en una imagen
def findFace(pathForImage):
    image = cv2.imread(pathForImage)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    haarCascadeFile = "C:/Users/kevin/Proyectos/models/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haarCascadeFile)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roiGray = gray[y:y+h, x:x+w]
        return roiGray  # devolvemos la primera cara encontrada

    return None  # si no encuentra cara

# 5) Función para preprocesar la ROI para el modelo
def prepareImageForModel(faceImage):
    resized = cv2.resize(faceImage, (48, 48), interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized, axis=0)
    imgResult = imgResult / 255.0
    return imgResult

# 6) Directorio y ruta de la imagen de prueba
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
testImagePath = os.path.join(BASE_DIR, "Hapyface.jpg")

# —–––––––––––––––––––––––––––––––
# DEBUG: Verificar rutas
print("BASE_DIR       :", BASE_DIR)
print("testImagePath  :", testImagePath)
print("Archivo existe?:", os.path.isfile(testImagePath))

# 7) Detección en imagen estática
faceGrayImage = findFace(testImagePath)
if faceGrayImage is None:
    print("No se encontró ninguna cara en la imagen.")
    exit()

imgForModel = prepareImageForModel(faceGrayImage)

# 8) Ejecución de la predicción
resultArray = model.predict(imgForModel, verbose=1)
answers = np.argmax(resultArray, axis=1)
idx = answers[0]

emotion_en = categories[idx]         # etiqueta en inglés
emotion_es = label_map[emotion_en]   # traducción al español

print("Predicted (en):", emotion_en)
print("Predicted (es):", emotion_es)

# 9) Mostrar la imagen con la emoción en español
img = cv2.imread(testImagePath)
font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img, emotion_es, (10, 25), font, 0.8, (209, 19, 77), 2)
cv2.imshow("Detección de Emoción", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
