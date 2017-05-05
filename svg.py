import xml.etree.ElementTree as ET
from PyQt4 import QtCore
import numpy as np

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

