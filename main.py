import cv2
import face_recognition
import numpy as np
from tensorflow.keras.models import load_model

#Load the trained model
model = load_model('emotion_recognition_model.keras')

#Decode
def decode_emotion(pred):
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[np.argmax(pred)]

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    #single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the image from BGR color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Extract the face region
        face_image = rgb_frame[top:bottom, left:right]
        face_image = cv2.resize(face_image, (48, 48)) # Resize to 48x48 pixels
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY) 
        face_image = np.expand_dims(face_image, axis=-1)  
        face_image = np.expand_dims(face_image, axis=0)

        # Predict emotion   
        predictions = model.predict(face_image)
        emotion = decode_emotion(predictions)

        # Display the emotion on the frame
        cv2.putText(frame, f"{emotion}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()