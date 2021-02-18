import cv2
import mediapipe as mp
from os import path
from utils import *
from PIL import ImageFont, ImageDraw, Image

def read_landmarks(multi_hand_landmarks):
    record = []
    if multi_hand_landmarks and len(multi_hand_landmarks)>0:
        for item in next(iter(multi_hand_landmarks)).landmark:
            record.extend([item.x, item.y, item.z])
    return record
 
def draw_text(frame, text, x, y, color= COLOR_GREEN, thickness=2, size=0.3):
    return cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.5, min_tracking_confidence=0.5)
cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
width = 1400
height = 768
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.cv2.CAP_PROP_FPS ,60)
records, labels = [], []

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image = cv2.flip(image, 1)
    # image = cv2.resize(image, (1024, 768))
    image = cv2.GaussianBlur(image, (5,5), 0)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        continue
    for hand_landmarks in results.multi_hand_landmarks:
        # print(
        #     f'Index finger tip coordinates: (',
        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        # )
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    record = read_landmarks(results.multi_hand_landmarks)
    label = None
    key_pressed = cv2.waitKey(5) & 0xFF
    if key_pressed == 27:
        break
    elif key_pressed == ord('a'):
        label = 1
    elif key_pressed == ord('s'):
        label = 0
    if label!=None:
        records.append(record)
        labels.append(label)
    pos_label_count = sum(labels)
    total_count = len(labels)
    neg_label_count  = total_count - pos_label_count
    image = draw_text(image, f'Total: {total_count}', 20,50, size=1.5)
    image = draw_text(image, f'POS: {pos_label_count}', 20,90, size=1.5)
    image = draw_text(image, f'NEG: {neg_label_count}', 20,130, size=1.5)
    cv2.imshow("preview", image)

hands.close()
cap.release()
cv2.destroyAllWindows()

if len(records)>0:
    num = 1
    f_name = path.join('pose_data', f'data_{num}.pickle')
    while path.exists(f_name):
        num += 1
        f_name = path.join('pose_data', f'data_{num}.pickle')
    dump_file_as_pickle({
        'data': records,
        'labels': labels
    },f_name)