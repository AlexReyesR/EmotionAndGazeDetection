import cv2
import numpy as np
import dlib
from imutils import face_utils
import face_recognition
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from gaze_tracking import GazeTracking
from datetime import datetime,date

USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path, compile=False)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('Webcam capture')
gaze = GazeTracking()

# Select video or webcam feed
print("Webcam selection for capture")
cap = cv2.VideoCapture(0) # Webcam source

save_time = datetime.time(datetime.now())
data_to_file = []

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()
    
    #bgr_image = video_capture.read()[1]


    gaze.refresh(bgr_image)

    bgr_image = gaze.annotated_frame()
    text = ""
    '''print("Horizontal is")
    print(gaze.horizontal_ratio())
    print("Vertical is")
    print(gaze.vertical_ratio())'''
    if gaze.is_right():
        text = "Looking mono chino de pelo morado"
    elif gaze.is_left():
        text = "Looking mona china"
    elif gaze.is_up():
        text = "Looking mono chino rubio intenso"
    elif gaze.is_down():
        text = "Looking logo"
    elif gaze.is_center():
        text = "Looking mono chino de pelo verde"

    cv2.putText(bgr_image, text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    #cv2.putText(bgr_image, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    #cv2.putText(bgr_image, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    #cv2.imshow("Demo", frame)

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_image)
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if (emotion_text and text):
            actual_time = datetime.time(datetime.now())

            # Save data each 3 seconds
            if ((datetime.combine(date.today(), actual_time) - datetime.combine(date.today(), save_time)).total_seconds() > 2):
                save_time = actual_time
                data_to_file.append([emotion_text, text])

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Webcam capture', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

data_file = open("data_file.txt", "w")

for data in data_to_file:
    data_file.write(str(data[0]) + " " + str(data[1]) + "\n")

data_file.close()

cap.release()
cv2.destroyAllWindows()
