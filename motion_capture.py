from PIL import Image, ImageDraw
import sys
import face_recognition
import cv2
import numpy as np
import math
from PyQt4 import QtGui, QtSvg, QtCore
import xml.etree.ElementTree as ET

class Attribute:
    name = None
    data = None

    def __init__(self, n, v):
        if n[0] == '{':
            self.name = n[n.find('}') + 1:]
        else:
            self.name = n
        self.data = v

    def blend(self, c):
        return self.name + '="'+self.data + '"'

    def merge(self, n, svg):
        assert self.name == svg.name
        assert self.data == svg.data

class Path:
    data = {}

    def __init__(self, d):
        self.data = d.split(' ')
        self.data = {'idle': [self.parse(x) for x in self.data]}
        print(self.data)

    def blend(self, coefs):
        res = 'd="'
        for i in range(len(self.data['idle'])):
            x = np.zeros((2,))
            is_text = False
            for coef_name, coef_value in coefs.iteritems():
                here = self.data[coef_name][i]
                if type(here) is str:
                    x = here
                    is_text = True
                    break
                x += coef_value * here
            if is_text:
                res += x + ' '
            else:
                res += str(x[0]) + ',' + str(x[1]) + ' '
        res += '"'
        return res

    def parse(self, d):
        if '0' <= d[0] <= '9':
            spl = d.split(',')
            return np.array([float(spl[0]), float(spl[1])])
        return d

    def merge(self, n, svg):
        self.data[n] = svg.data['idle']

class Node:
    tag = None
    attributes = {}
    children = []

    def __init__(self, t, a, c):
        if t[0] == '{':
            self.tag = t[t.find('}') + 1:]
        else:
            self.tag = t
        self.attributes = a
        self.children = c

    def blend(self, c):
        res = '<' + self.tag + " " + " ".join([v.blend(c) for k, v in self.attributes.iteritems()])
        if len(self.children) == 0:
            res += ' />'
        else:
            res += '>'
            res += ''.join([ch.blend(c) for ch in self.children])
            res +='</' + self.tag + '>'
        return res

    def merge(self, n, svg):
        assert self.tag == svg.tag
        print(self.tag, svg.tag, len(self.attributes), len(svg.attributes))
        assert len(self.attributes) == len(svg.attributes)
        assert len(self.children) == len(svg.children)

        for k, v in self.attributes.iteritems():
            self.attributes[k].merge(n, svg.attributes[k])

        for i in range(len(self.children)):
            self.children[i].merge(n, svg.children[i])

class Text:
    data = None

    def __init__(self, d):
        self.data = d

    def merge(self, n, svg):
        assert self.data == svg.data

    def blend(self, c):
        return self.data

class Svg:
    root = None
    def __init__(self, filename):
        self.root = self.parse(ET.parse(filename).getroot())

    def parse(self, xml):
        c = None
        if xml.text and xml.text.strip('\n\r ') != "":
            print(repr(xml.text))
            c = [Text(xml.text)]
        else:
            c = [self.parse(x) for x in xml]
        return Node(xml.tag,
                {n: self.parseAttr(n, a) for n, a in xml.attrib.iteritems()},
                c)

    def parseAttr(self, n, v):
        if n == 'd':
            return Path(v)
        return Attribute(n, v)

    def merge(self, n, svg):
        self.root.merge(n, svg.root)

    def blend(self, c):
        return QtCore.QByteArray('<?xml version="1.0" encoding="UTF-8" standalone="no"?>'+ self.root.blend(c))

svg = Svg('squarish.svg')
svg.merge('d2', Svg('squarish2.svg'))

app = QtGui.QApplication(sys.argv)
svgWidget = QtSvg.QSvgWidget()
svgWidget.setGeometry(50,50,759,668)
svgWidget.show()
#svgWidget.load('drawing.svg')
svgWidget.load(svg.blend({'idle': 1}))

video_capture = cv2.VideoCapture(0)

def to_pts(xs):
    return [np.array(xs, dtype=int) * 2]

class Landmarks:
    lds = None

    def __init__(self, xs):
        self.lds = xs

    def face_center(self):
        return self.lds['nose_bridge'][0]

    def get_angle(self):
        bridge = to_pts(self.lds['nose_bridge'])
        angle = bridge[0][-1] - bridge[0][0]
        angle = math.atan2(angle[1], angle[0]) - math.pi / 2
        return angle

    def landmarks_map(self, f):
        return Landmarks({ k: [f(v0) for v0 in v] for k, v in self.lds.iteritems()})

    def translate_point(self, x, tr):
        return [x[0] + tr[0], x[1] + tr[1]]

    def translate(self, tr):
        return self.landmarks_map(lambda x: self.translate_point(x, tr))

    def rotate_point(self, x, center, angle):
        x = [x[0] - center[0], x[1] - center[1]]
        return [x[0] * math.cos(angle) - x[1] * math.sin(angle) + center[0],
                x[0] * math.sin(angle) + x[1] * math.cos(angle) + center[1]]

    def scale(self, s):
        return self.landmarks_map(lambda x: [x[0] * s, x[1] * s])

    def unscale(self):
        bridge = to_pts(self.lds['nose_bridge'])
        length = np.linalg.norm(bridge[0][-1] - bridge[0][0])
        return self.scale(1 / length)

    def rotate(self, angle):
        center = self.face_center()
        return self.landmarks_map(lambda x: self.rotate_point(x, center, angle))

    def normalize(self):
        angle = self.get_angle()
        center = self.face_center()
        return self.rotate(-angle).translate([-center[0], -center[1]]).unscale()

    def fingerprint(self):
        return np.array(self.lds['bottom_lip'] + self.lds['top_lip'])

still_print = None
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    face_landmarks_list = face_recognition.face_landmarks(small_frame)

    for face_landmarks in face_landmarks_list:
        face_landmarks = Landmarks(face_landmarks)
        face_landmarks = face_landmarks.normalize().scale(100).translate([150, 40])
        still_print = face_landmarks.fingerprint()
        face_landmarks = face_landmarks.lds
        #cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        # Make the eyebrows into a nightmare
        cv2.fillPoly(frame, to_pts(face_landmarks['left_eyebrow']), (68, 54, 39, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['right_eyebrow']), (68, 54, 39, 128))

        cv2.fillPoly(frame, to_pts(face_landmarks['left_eye']), (150, 54, 39, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['right_eye']), (150, 54, 39, 128))

        # Gloss the lips
        cv2.fillPoly(frame, to_pts(face_landmarks['top_lip']), (150, 0, 0, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['bottom_lip']), (150, 0, 0, 128))


        cv2.fillPoly(frame, to_pts(face_landmarks['nose_bridge']), (150, 0, 0, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['nose_tip']), (150, 0, 0, 128))
        cv2.polylines(frame, to_pts(face_landmarks['chin']), False, (150, 0, 0, 128))
    cv2.imshow('Vid', frame)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

print(still_print)

omouth_print = None
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    face_landmarks_list = face_recognition.face_landmarks(small_frame)

    for face_landmarks in face_landmarks_list:
        face_landmarks = Landmarks(face_landmarks)
        face_landmarks = face_landmarks.normalize().scale(100).translate([150, 40])
        omouth_print = face_landmarks.fingerprint()
        face_landmarks = face_landmarks.lds
        #cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        # Make the eyebrows into a nightmare
        cv2.fillPoly(frame, to_pts(face_landmarks['left_eyebrow']), (68, 54, 39, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['right_eyebrow']), (68, 54, 39, 128))

        cv2.fillPoly(frame, to_pts(face_landmarks['left_eye']), (150, 54, 39, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['right_eye']), (150, 54, 39, 128))

        # Gloss the lips
        cv2.fillPoly(frame, to_pts(face_landmarks['top_lip']), (150, 0, 0, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['bottom_lip']), (150, 0, 0, 128))


        cv2.fillPoly(frame, to_pts(face_landmarks['nose_bridge']), (150, 0, 0, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['nose_tip']), (150, 0, 0, 128))
        cv2.polylines(frame, to_pts(face_landmarks['chin']), False, (150, 0, 0, 128))
    cv2.imshow('Vid', frame)

    if cv2.waitKey(1) & 0xFF == ord('z'):
        break

print(omouth_print)

def similarity(lds):
    st = np.exp(-np.linalg.norm(lds - still_print)  / 20)
    om = np.exp(-np.linalg.norm(lds - omouth_print) / 20)
    total = st + om
    print(st / total, om / total)
    return {'idle': st / total, 'd2': om / total }


while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    face_landmarks_list = face_recognition.face_landmarks(small_frame)

    for face_landmarks in face_landmarks_list:
        face_landmarks = Landmarks(face_landmarks)
        face_landmarks = face_landmarks.normalize().scale(100).translate([150, 40])
        weights = similarity(face_landmarks.fingerprint())
        svgWidget.load(svg.blend(weights))
        print(weights)
        face_landmarks = face_landmarks.lds
        #cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        # Make the eyebrows into a nightmare
        cv2.fillPoly(frame, to_pts(face_landmarks['left_eyebrow']), (68, 54, 39, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['right_eyebrow']), (68, 54, 39, 128))

        cv2.fillPoly(frame, to_pts(face_landmarks['left_eye']), (150, 54, 39, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['right_eye']), (150, 54, 39, 128))

        # Gloss the lips
        cv2.fillPoly(frame, to_pts(face_landmarks['top_lip']), (150, 0, 0, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['bottom_lip']), (150, 0, 0, 128))


        cv2.fillPoly(frame, to_pts(face_landmarks['nose_bridge']), (150, 0, 0, 128))
        cv2.fillPoly(frame, to_pts(face_landmarks['nose_tip']), (150, 0, 0, 128))
        cv2.polylines(frame, to_pts(face_landmarks['chin']), False, (150, 0, 0, 128))
    cv2.imshow('Vid', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
