from PIL import Image, ImageDraw
import sys
import face_recognition
import cv2
import numpy as np
import math
from svg import Svg
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

class Landmarks:
    #landmarks are of the shape (N, 2)
    lds = None

    def __init__(self, xs):
        if not type(xs) is np.array:
            self.lds = {k: np.array(v) for k, v in xs.iteritems()}
        else:
            self.lds = xs

    def face_center(self):
        return self.lds['nose_bridge'][0,:].reshape(1, 2)

    def get_angle(self):
        bridge = self.lds['nose_bridge']
        angle = bridge[-1,:] - bridge[0,:]
        angle = math.atan2(angle[1], angle[0]) - math.pi / 2
        return angle

    def translate(self, tr):
        for k, v in self.lds.iteritems():
            self.lds[k] += tr

    def scale(self, s):
        for k, v in self.lds.iteritems():
            self.lds[k] *= s

    def unscale(self):
        bridge = self.lds['nose_bridge']
        length = np.linalg.norm(bridge[-1,:] - bridge[0,:])
        return self.scale(1 / length)

    def rotate(self, angle):
        center = self.face_center()
        rot = np.array(
                [[math.cos(angle), math.sin(angle)],
                 [-math.sin(angle), math.cos(angle)]])
        for k, v in self.lds.iteritems():
            self.lds[k] = (v - center).dot(rot) + center

    def normalize(self):
        angle = self.get_angle()
        center = self.face_center()
        self.rotate(-angle)
        self.translate(-center)
        self.unscale()

    def fingerprint(self):
        return np.array(self.lds['bottom_lip'] + self.lds['top_lip'])

    def get(self, part):
        return self.lds[part].astype(dtype=int)

    def merge_with_new_frame(self, lds, damp=0.7):
        for k, v in self.lds.iteritems():
            self.lds[k] = self.lds[k] * damp + lds.lds[k] * (1 - damp)


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
