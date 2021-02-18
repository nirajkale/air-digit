import cv2
import mediapipe as mp
from os import path
from utils import *
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf

CAMERA_DISTANCE = -0.1
INTERPOLATION_THRESHOLD_PER = 25

COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
 
def draw_text(frame, text, x, y, color= COLOR_GREEN, thickness=2, size=0.3):
    return cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def read_landmarks(multi_hand_landmarks):
    record = []
    if multi_hand_landmarks and len(multi_hand_landmarks)>0:
        for item in next(iter(multi_hand_landmarks)).landmark:
            record.extend([item.x, item.y, item.z])
    return record

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.5, min_tracking_confidence=0.5)
cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
width = 768
height = 768
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.cv2.CAP_PROP_FPS ,30)
indexed_collection = []
print('loading mnist model..')
model_mnist = tf.keras.models.load_model(r'saved_models/mnist.h5')
# model_pose = tf.keras.models.load_model(r'saved_models/pose.h5')

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    H, W, _ = image.shape
    image = cv2.flip(image, 1)
    # image = cv2.GaussianBlur(image, (5,5), 0)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    index = None
    pose_detected = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            selection_indices = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,mp_hands.HandLandmark.RING_FINGER_TIP]
            fingers = [(hand_landmarks.landmark[i].y, hand_landmarks.landmark[i]) for i in selection_indices]
            fingers = sorted(fingers, key=lambda item: item[0], reverse=False)
            index = fingers[0][1]
            break
    key_pressed = cv2.waitKey(5) & 0xFF
    if key_pressed == 27:
        break
    elif key_pressed == ord('a'):
        indexed_collection = []
    if index:
        if index.z < CAMERA_DISTANCE:
            indexed_collection.append((index.x , index.y ))
            image = draw_text(image, f'Z: {round(index.z,2)}', 20,130, size=1.5, color= COLOR_RED)
        else:
            image = draw_text(image, f'Z: {round(index.z,2)}', 20,130, size=1.5, color= COLOR_GREEN)
    # record = read_landmarks(results.multi_hand_landmarks)
    # if len(record)>0:
    #     x = np.reshape(np.array(record), (1, len(record)))
    #     preds = model_pose.predict(x, verbose=False)
    #     if preds.item()>0.5 or (index!=None and index.z < CAMERA_DISTANCE):
    #         indexed_collection.append((index.x , index.y ))
    #         pose_detected = True
    blank_image = image.copy()
    blank_image[:,:,:]=0
    if len(indexed_collection)>0:
        x2, y2 = None, None
        for (x1,y1) in indexed_collection:
            x1, y1 = x1* W, y1* H
            x3, y3 = x1* 28, y1* 28
            image = cv2.circle(image, (int(x1),int(y1)), radius=5, color=COLOR_RED, thickness=-1)
            blank_image = cv2.circle(blank_image, (int(x1),int(y1)), radius=12, color=COLOR_WHITE, thickness=-1)
            pass
            if x2:
                dist = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
                if dist < INTERPOLATION_THRESHOLD_PER:
                    image = cv2.line(image, (int(x2), int(y2)), (int(x1), int(y1)), COLOR_RED, thickness=10) 
                    blank_image = cv2.line(blank_image, (int(x2), int(y2)), (int(x1), int(y1)), COLOR_WHITE, thickness=12) 
                    pass
            x2, y2 = x1, y1
    blank_image = cv2.resize(blank_image, (28, 28),  interpolation = cv2.INTER_NEAREST)
    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    blank_image = np.reshape(blank_image, (1,28,28,1))
    preds = model_mnist.predict(blank_image, verbose=False)
    digit = int(preds[0].argmax())
    image = draw_text(image, f'Digit : {digit}', 20,50, size=1.5)
    cv2.imshow("preview", image)

hands.close()
cap.release()
cv2.destroyAllWindows()