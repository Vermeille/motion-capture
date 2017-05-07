import numpy as np
import math

class Landmarks:
    #landmarks are of the shape (N, 2)
    lds = None

    def __init__(self, xs):
        if not type(xs) is np.array:
            self.lds = {k: np.array(v) for k, v in xs.iteritems()}
        else:
            self.lds = xs

    def face_center(self):
        return self.lds['nose_tip'][0,:].reshape(1, 2)

    def get_angle(self):
        bridge = self.lds['nose_tip']
        angle = bridge[-1,:] - bridge[0,:]
        angle = math.atan2(angle[1], angle[0])
        return angle

    def translate(self, tr):
        for k, v in self.lds.iteritems():
            self.lds[k] += tr

    def scale(self, s):
        for k, v in self.lds.iteritems():
            self.lds[k] *= s

    def unscale(self):
        bridge = self.lds['nose_tip']
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

