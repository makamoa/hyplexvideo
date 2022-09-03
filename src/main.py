#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tkinter
from tkCamera import tkCamera
from utils import Camera


"""TODO: add docstring"""


HOME = os.path.dirname(os.path.abspath(__file__))


class App:
    def __init__(self, parent, camera, title, sources):
        """TODO: add docstring"""

        self.parent = parent

        self.parent.title(title)

        self.stream_widgets = []

        width = 400
        height = 300

        columns = 2
        for number, (text, source) in enumerate(sources):
            widget = tkCamera(self.parent, camera, text, source, width, height, sources)
            row = number // columns
            col = number % columns
            widget.grid(row=row, column=col)
            self.stream_widgets.append(widget)

        self.parent.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self, event=None):
        """TODO: add docstring"""

        print("[App] stoping threads")
        for widget in self.stream_widgets:
            widget.vid.running = False

        print("[App] exit")
        self.parent.destroy()


if __name__ == "__main__":

    sources = [  # (text, source)
        # local webcams
        (
            "raw output",
            "/home/makam0a/Dropbox/Pixeltra/Datasets/Videos/channels/fake2-raw-output-4.avi",
        ),
        # remote videos (or streams)
        # (
        #     "channels",
        #     "/home/makam0a/Dropbox/Pixeltra/Datasets/Videos/channels/fake2-output-3.avi",
        # ),
        # ("spectra", "/home/makam0a/Dropbox/Pixeltra/Datasets/Videos/channels/fake2-output-4.avi"),
        # (
        #     "segmentation mask",
        #     "/home/makam0a/Dropbox/Pixeltra/Datasets/Videos/channels/fake2-output-2.avi",
        # ),
        # local files
        # ('2021.01.25 20.37.50.avi', '2021.01.25 20.37.50.avi'),
    ]

    root = tkinter.Tk()
    camera = Camera()
    App(root, camera, "HOCULUS", sources)
    root.mainloop()
