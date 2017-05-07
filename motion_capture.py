from PIL import Image, ImageDraw
import sys
import scipy.optimize
import face_recognition
import cv2
import numpy as np
import math
import sys
from svg import Svg
from landmarks import Landmarks
from PyQt4 import QtGui, QtSvg, QtCore

if len(sys.argv) < 2:
    sys.stderr.write("usage: motion_capture.py pose1.svg pose2.svg...\n")
    sys.exit(0)

frame_subsampling_factor = 2

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
    cv2.imshow('Vid', frame)

    while not lds:
        frame, lds = one_frame()
        cv2.imshow('Vid', frame)

    while True:
        if cv2.waitKey(1) & 0xFF == ord(stop):
            break

        frame, lds2 = one_frame()

        if lds2:
            lds.merge_with_new_frame(lds2)
            draw_landmarks(frame, lds)

        cv2.imshow('Vid', frame)
    return lds.fingerprint()

app = QtGui.QApplication(sys.argv)
svgWidget = QtSvg.QSvgWidget()
svgWidget.setGeometry(50,50,759,668)
svgWidget.show()

video_capture = cv2.VideoCapture(0)

faces = {}
svg = Svg(sys.argv[1])
for fname in sys.argv[1:]:
    svg2 = Svg(fname)
    svg.merge(fname, svg2)

    svgWidget.load(svg.blend({fname: 1}))
    face = capture_landmarks(stop='a')
    faces[fname] = face
print(faces)

def softmax(xs):
    xs = np.exp(xs)
    xs /= np.sum(xs)
    return xs

def apply_weights(ws, faces):
    total_face = 0
    ws_i = 0
    for k in sorted(faces):
        total_face = faces[k] * ws[ws_i] + total_face
        ws_i += 1
    return total_face

def reconstruction_loss(real_face, reconstructed_face):
    diff = (real_face - reconstructed_face) ** 2
    return diff.sum()

def sparsity_loss(ws):
    return np.abs(ws).sum()

def compare_faces(ws, faces, cur_face):
    ws = softmax(ws)
    rec_face = apply_weights(ws, faces)
    return reconstruction_loss(cur_face, rec_face) + 0.2 * sparsity_loss(ws)

def find_weights(faces, cur_face):
    ws = np.zeros((len(faces),))
    ws[0] = 2
    solution, iters, rc = scipy.optimize.fmin_tnc(compare_faces, ws, args=(faces, cur_face),
            approx_grad=True, epsilon=1e-5)
    solution = softmax(solution)
    ws = {}
    i = 0
    for k in sorted(faces):
        ws[k] = solution[i]
        i += 1
    return ws

def similarity(lds, faces):
    similarities = {}
    total = 0
    for k, v in faces.iteritems():
        sim = np.exp(-np.linalg.norm(lds - v) / 40)
        similarities[k] = sim
        total += sim
    for k in similarities:
        similarities[k] /= total
    return similarities

frame, lds = one_frame()
while True:
    frame, lds2 = one_frame()
    if lds2:
        lds.merge_with_new_frame(lds2)
        #similarities = similarity(lds.fingerprint(), faces)
        similarities = find_weights(faces, lds.fingerprint())
        draw_landmarks(frame, lds)
        svgWidget.load(svg.blend(similarities))

    cv2.imshow('Vid', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
