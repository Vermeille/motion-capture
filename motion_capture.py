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

def capture_landmarks(stop=' '):
    lds = None
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        face_landmarks_list = face_recognition.face_landmarks(small_frame)

        for face_landmarks in face_landmarks_list:
            face_landmarks = Landmarks(face_landmarks)
            face_landmarks.normalize()
            face_landmarks.scale(100)
            face_landmarks.translate(np.array([[150, 40]]))
            draw_landmarks(frame, face_landmarks)
            lds = face_landmarks.fingerprint()
        cv2.imshow('Vid', frame)

        if cv2.waitKey(1) & 0xFF == ord(stop):
            break
    return lds


still_print = capture_landmarks(stop='a')
print(still_print)

omouth_print = capture_landmarks(stop='z')
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
        face_landmarks.normalize()
        face_landmarks.scale(100)
        face_landmarks.translate([150, 40])
        weights = similarity(face_landmarks.fingerprint())
        svgWidget.load(svg.blend(weights))
        print(weights)
        draw_landmarks(frame, face_landmarks)
    cv2.imshow('Vid', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
