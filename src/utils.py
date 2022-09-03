import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rotate
import matplotlib.colors as mcolors
from scipy import ndimage
import pickle, os

def stack_images(image):
    wx, wy, ch = image.shape
    cols = []
    for i in range(3):
        col = np.concatenate([image[:,:,i],image[:,:,i+1],image[:,:,i+2]])
        cols.append(col)
    return np.concatenate(cols,axis=1)

white_alignment = {
    'file' : 'alignment.bmp',
    'rotate' : 1.0,
    'x0' : 511,
    'y0' : 369,
    'step' : 7.287 * 3,
    'xmax' : 1360,
    'xmin' : 100,
    'ymax' : 1150,
    'ymin' : 65
}

fake_alignment = {
    'file' : 'fake1.bmp',
    'rotate' : 0.0,
    'x0' : 512,
    'y0' : 393,
    'step' : 7.287 * 3,
    'xmax' : 1360,
    'xmin' : 100,
    'ymax' : 1155,
    'ymin' : 65
}

real_alignment = {
    'file' : 'real.bmp',
    'rotate' : 0.0,
    'x0' : 512,
    'y0' : 394,
    'step' : 7.287 * 3,
    'xmax' : 1360,
    'xmin' : 100,
    'ymax' : 1155,
    'ymin' : 68
}

class Camera():
    def __init__(self, alignment=real_alignment, nx=48, ny=56, channels=34,
                 datadir='/home/makam0a/Dropbox/projects/camera/data/'):
        self.alignment = alignment
        self.nx = nx
        self.ny = ny
        self.channels = channels
        self.shape = (nx, ny, channels)
        self.wl = np.linspace(400,730,channels)
        self.mean_ = np.load(os.path.join(datadir,'pca_mean.npy'))
        self.filters = np.load(os.path.join(datadir,'filters.npy'), allow_pickle=True)
        self.align()

    def align(self):
        x0 = self.alignment['x0']
        y0 = self.alignment['y0']
        step = self.alignment['step']
        xmax = self.alignment['xmax']
        xmin = self.alignment['xmin']
        ymax = self.alignment['ymax']
        ymin = self.alignment['ymin']
        nx = self.nx
        ny = self.ny
        output = np.zeros((nx, ny, 9))
        self.xx = []
        self.yy = []
        for i in range(3):
            for j in range(3):
                x = []
                y = []
                px = np.round(np.arange(x0, xmax, step) + i * step / 3).astype(
                    np.int)
                py = np.round(np.arange(y0, ymax, step) + j * step / 3).astype(
                    np.int)
                x += [px[(px > xmin) & (px < xmax)]]
                y += [py[(py > ymin) & (py < ymax)]]
                x = np.concatenate(x, axis=0)
                y = np.concatenate(y, axis=0)
                xx, yy = np.meshgrid(x, y)
                self.xx.append(xx)
                self.yy.append(yy)

    def draw_alignment(self, ax=None):
        if ax is None:
            ax = plt.gca()
        for i in range(3):
            for j in range(3):
                ax.plot(self.xx[3*i + j], self.yy[3*i + j], marker='.', color=mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[(3 * i + j) % 9]],
                         linestyle='none')

    def show_alignment(self, img, wx=None, wy=None, figsize=(20,20), **fig_kw):
        self.image = self.preprocess(img)
        nx, ny = img.shape[:2]
        if wx is None:
            wx = nx // 3
        if wy is None:
            wy = ny // 3
        xc = np.linspace(wx/2, nx - wx/2,3)
        yc = np.linspace(wy/2, ny - wy/2,3)
        fig, axes = plt.subplots(3,3, figsize=figsize)
        for i, axx in enumerate(axes):
            for j, ax in enumerate(axx):
                ax.set_xlim(xc[j] - wx // 2, xc[j] + wx // 2)
                ax.set_ylim(yc[i] - wy // 2, yc[i] + wy // 2)
                ax.axvline(xc[j])
                ax.axhline(yc[i])
                ax.axis('off')
                ax.imshow(self.image, cmap='hot')
                self.draw_alignment(ax)
        fig.tight_layout()
        fig.show()
        plt.figure(figsize=(20,20))
        plt.imshow(self.image, cmap='hot')
        self.draw_alignment()
        plt.show()

    def preprocess(self, img, filter_size=1):
        rotation = self.alignment['rotate']
        img = rotate(img, rotation)
        filter = np.full((filter_size, filter_size), 1 / filter_size ** 2)
        img = ndimage.convolve(img, filter)
        return img

    def transform(self, image):
        image = self.preprocess(image)
        output = np.zeros((self.nx, self.ny, 9))
        for i in range(9):
            output[:, :, i] = image[self.yy[i], self.xx[i]]
        return output

    def inverse(self, encoded_image):
        shape = encoded_image.shape[:2]
        data_original = np.dot(encoded_image, self.filters) + self.mean_
        return data_original.reshape((*shape, self.channels))