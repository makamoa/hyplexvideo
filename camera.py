import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

class MetaSurface():
    def __init__(self):
        pass

class MetaCamera():
    def __init__(self):
        pass

    def calibration(self):
        pass

    def align(self):
        pass

    def display_alignment(self):
        pass

class MetaPixel():
    def __init__(self, start, npixels=3, size=1):
        self.pixel_size = size
        self.size = size * npixels
        self.start_x, self.start_y = start
        self.npixels = npixels

    def get_pixel_coords(self):
        xx = np.arange(self.start_x, self.start_x + self.size * self.npixels, self.size)
        yy = np.arange(self.start_y, self.start_y + self.size * self.npixels, self.size)


    def show(self):
        pass


