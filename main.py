from tensorflow.keras.models import load_model
from ultralytics import YOLO
import numpy as np
import cv2



yolo_model = YOLO('yolov11s-face.pt') 
emotion_model = load_model('emotion_recognition_model.h5')

def preprocess_face(face):
    face = cv2.resize(face, (224, 224)) 
    face = face / 255.0  
    face = face.reshape(1, 224, 224, 3)
    return face


cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()  
    if not ret:
        break


    results = yolo_model(frame)

    for result in results:
        for box in result.boxes.xyxy:  
            x1, y1, x2, y2 = map(int, box)  
            face = frame[y1:y2, x1:x2]  

            if face.size > 0:
                face_preprocessed = preprocess_face(face)
                emotion_probs = emotion_model.predict(face_preprocessed, verbose=0)
                emotion_label = np.argmax(emotion_probs) 

                emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                emotion_text = emotion_classes[emotion_label]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()