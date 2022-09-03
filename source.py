# Maksim Makarenko
# Unsupervised segmentation of SEM images
# 20.04.2021
import skimage
# import numpy as np
# from skimage import io
# import os, re
# import matplotlib.pyplot as plt
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.preprocessing import OneHotEncoder
# import scipy.ndimage as ndi
# from skimage import data
# from skimage.exposure import histogram
# from skimage import color, segmentation
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import histogram
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage import data
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter

def random_slice(image_shape, region_size=200):
    if not hasattr(region_size, '__getitem__'):
        # if 1D size, copy to sceond dimension
        region_size = [region_size]*2
    wx, wy = image_shape[:2]
    b = (wx - region_size[0]) - 1
    cx = np.random.randint(0,b)
    b = (wx - region_size[1]) - 1
    cy = np.random.randint(0,b)
    return (slice(cx,cx+region_size[0]),slice(cy,cy+region_size[1]))

class SemSegmentation():
    def __init__(self, sigma=3, convex=False , th=None, classes=3):
        """
        :param sigma:
        :param convex:
        :param th:
        :param classes:
        """
        if th is None:
            th = 1 / sigma ** 2
        self.sigma = sigma
        self.th = th
        self.convex = convex
        self.classes = classes

    def fit(self, image):
        """
        :param image:
        :return:
        """
        # create markers using grayscale threshold
        self.image = image
        thresholds = threshold_multiotsu(image, classes=self.classes)
        markers = np.digitize(image, bins=thresholds)
        nsamples_in_cluster = np.array([markers[markers == cluster].__len__() for cluster in np.unique(markers)])
        cluster = nsamples_in_cluster.argmin()
        mask = markers == cluster
        markers[mask] = 1
        markers[~mask] = 0
        self.markers = markers
        # filter markers with gaussian kernel. > features size > kernel.
        filtered = gaussian_filter(markers.astype(float), sigma=self.sigma)
        filtered[filtered < self.th] = 0
        filtered[filtered > + self.th] = 1
        labels = skimage.measure.label(filtered.astype(np.int64),background=0)
        segments = filtered
        if self.convex:
            segments, labels = self.convex_transform(segments, labels)
        # sometimes it could be better to use convex_hull of a segment
        self.segments = segments
        self.labels = labels
        self.regions = regionprops(labels)
        self.eval_summary()

    def fit_predict(self, *pargs, **kargs):
        self.fit(*pargs, **kargs)
        return self.segments

    def convex_transform(self, segments, labels):
        """
        :param segments:
        :param labels:
        :return:
        """
        props = regionprops(labels)
        for region in props:
            minr, minc, maxr, maxc = region.bbox
            segments[minr:maxr, minc:maxc] = region.convex_image
        labels = skimage.measure.label(segments.astype(np.int64), background=0)
        return segments, labels

    def show_colorized(self, ax=None, mask_type='segments'):
        """
        :param ax:
        :param region:
        :return:
        """
        if ax is None:
            ax = plt.gca()
        mask = self.__getattribute__(mask_type)
        regions_colorized = label2rgb(mask, image=self.image, bg_label=0)
        ax.imshow(regions_colorized)
        ax.set_axis_off()

    def eval_centroid(self, label):
        return ndi.measurements.center_of_mass(self.segments, self.labels, label)

    def eval_summary(self):
        self.n_segments = len(np.unique(self.labels[1:]))
        self.seg_areas = np.array([region.area for region in self.regions])
        self.mu_area = self.seg_areas.mean()
        self.sigma_area = self.seg_areas.std()
        self.centroids = np.array([region.centroid for region in self.regions])

    def add_summary(self, ax=None, slices=None, text_shift=20, fontsize=30, show_centroids=True):
        if ax is None:
            ax = plt.gca()
        for (cx, cy), area in zip(self.centroids,self.seg_areas):
            if slices is not None:
                xs, ys = slices[:2]
                if (cx < xs.start or cx > xs.stop) or (cy < ys.start or cy > ys.stop):
                    continue
                else:
                    cx-=xs.start
                    cy-=ys.start
            if show_centroids:
                ax.plot([cy],[cx],'ob',lw=50)
            ax.text(cy,cx+text_shift,"area = %d" % area, fontsize=fontsize)

    def show(self, fig=None, with_summary=False, summary_kw={}):
        """
        :param fig:
        :return:
        """
        if fig is None:
            fig = plt.gcf()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(self.image, cmap='gray')
        ax1.set_axis_off()
        fig.suptitle("Number of segments = %d with average size = %.1f and std = %.1f"
                     % (self.n_segments, self.mu_area, self.sigma_area))
        self.show_colorized(ax=ax2, mask_type='segments')
        if with_summary:
            self.add_summary(ax2, **summary_kw)

    def show_crop(self, slices,
                  ax=None,
                  mask_type='segments',
                  with_summary=True,
                  show_centroids=True):
        if ax is None:
            ax = plt.gca()
        mask = self.__getattribute__(mask_type)
        regions_colorized = label2rgb(mask, image=self.image, bg_label=0)
        ax.imshow(regions_colorized[slices])
        ax.set_axis_off()
        if with_summary:
            self.add_summary(ax=ax,slices=slices,show_centroids=show_centroids)

    def show_random_crops(self, fig=None, grid=(2,3), crop_size=200, **kw):
        if fig is None:
            fig = plt.gcf()
        axes = fig.subplots(*grid)
        for ax in axes.flatten():
            self.show_crop(random_slice(self.image.shape, crop_size),ax=ax,**kw)

    def show_region(self, label=0, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.set_title('Region %d' % label)
        region = self.regions[label]
        minr, minc, maxr, maxc = region.bbox
        image_crop = self.image[minr:maxr, minc:maxc]
        region_colorized = label2rgb(region.image, image=image_crop, bg_label=0)
        ax.imshow(region_colorized)
        ax.set_axis_off()




