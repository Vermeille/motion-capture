from PIL import Image, ImageDraw
import sys
import face_recognition
import cv2
import numpy as np
import math
from svg import Svg
from landmarks import Landmarks
from PyQt4 import QtGui, QtSvg, QtCore

frame_subsampling_factor = 2

svg = Svg('squarish.svg')
svg.merge('d2', Svg('squarish2.svg'))

app = QtGui.QApplication(sys.argv)
svgWidget = QtSvg.QSvgWidget()
svgWidget.setGeometry(50,50,759,668)
svgWidget.show()
svgWidget.load(svg.blend({'idle': 1}))

video_capture = cv2.VideoCapture(0)

def draw_landmarks(frame, lds):
    #cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    # Make the eyebrows into a nightmare
    cv2.fillPoly(frame, [lds.get('left_eyebrow')], (68, 54, 39, 128))
    cv2.fillPoly(frame, [lds.get('right_eyebrow')], (68, 54, 39, 128))

    cv2.fillPoly(frame, [lds.get('left_eye')], (150, 54, 39, 128))
    cv2.fillPoly(frame, [lds.get('right_eye')], (150, 54, 39, 128))

    # Gloss the lips
    cv2.fillPoly(frame, [lds.get('top_lip')], (150, 0, 0, 128))
    cv2.fillPoly(frame, [lds.get('bottom_lip')], (150, 0, 0, 128))

    cv2.fillPoly(frame, [lds.get('nose_bridge')], (150, 0, 0, 128))
    cv2.fillPoly(frame, [lds.get('nose_tip')], (150, 0, 0, 128))
    cv2.polylines(frame, [lds.get('chin')], False, (150, 0, 0, 128))

def one_frame():
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    face_landmarks_list = face_recognition.face_landmarks(small_frame)

    for face_landmarks in face_landmarks_list:
        face_landmarks = Landmarks(face_landmarks)
        face_landmarks.normalize()
        face_landmarks.scale(100)
        face_landmarks.translate(np.array([[150, 40]]))
        return frame, face_landmarks
    return frame, None

def capture_landmarks(stop=' '):
    frame, lds = one_frame()

    while not lds:
        frame, lds = one_frame()

    while True:
        if cv2.waitKey(1) & 0xFF == ord(stop):
            break

        frame, lds2 = one_frame()

        if lds2:
            lds.merge_with_new_frame(lds2)
            draw_landmarks(frame, lds)

        cv2.imshow('Vid', frame)
    return lds.fingerprint()


still_print = capture_landmarks(stop='a')
print(still_print)

omouth_print = capture_landmarks(stop='z')
print(omouth_print)

def similarity(lds):
    st = np.exp(-np.linalg.norm(lds - still_print)  / 40)
    om = np.exp(-np.linalg.norm(lds - omouth_print) / 40)
    total = st + om
    print(st / total, om / total)
    return {'idle': st / total, 'd2': om / total }


frame, lds = one_frame()

while not lds:
    frame, lds = one_frame()

while True:
    frame, lds2 = one_frame()
    if lds2:
        lds.merge_with_new_frame(lds2)
        draw_landmarks(frame, lds)
        weights = similarity(lds.fingerprint())
        svgWidget.load(svg.blend(weights))

    cv2.imshow('Vid', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
