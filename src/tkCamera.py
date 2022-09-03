#!/usr/bin/env python

import PIL.ImageTk
import PIL.Image
import tkinter
import tkinter.filedialog
from videocapture import VideoCapture
from tkinter import ttk
import cv2
import numpy as np

"""TODO: add docstring"""


class tkSourceSelect(tkinter.Toplevel):

    def __init__(self, parent, other_sources=None):
        """TODO: add docstring"""

        super().__init__(parent)

        self.other_sources = other_sources

        # default values at start
        self.item = None
        self.name = None
        self.source = None

        # GUI
        button = tkinter.Button(self, text="Open file...", command=self.on_select_file)
        button.pack(fill='both', expand=True)

        if self.other_sources:
            tkinter.Label(self, text="Other Sources:").pack(fill='both', expand=True)

            for item in self.other_sources:
                text, source = item
                button = tkinter.Button(self, text=text, command=lambda data=item:self.on_select_other(data))
                button.pack(fill='both', expand=True)

    def on_select_file(self):
        """TODO: add docstring"""

        result = tkinter.filedialog.askopenfilename(
                                        initialdir=".",
                                        title="Select video file",
                                        filetypes=(("AVI files", "*.avi"), ("MP4 files","*.mp4"), ("all files","*.*"))
                                    )

        if result:
            self.item = item
            self.name = name
            self.source = source

            print('[tkSourceSelect] selected:', name, source)

            self.destroy()

    def on_select_other(self, item):
        """TODO: add docstring"""

        name, source = item

        self.item = item
        self.name = name
        self.source = source

        print('[tkSourceSelect] selected:', name, source)

        self.destroy()

class RawCanvas(tkinter.Frame):
    def __init__(self, parent, text="Raw Camera Image", width=None, height=None):
        super().__init__(parent)
        self.width  = width
        self.height = height
        self.label = tkinter.Label(self, text=text)
        self.label.pack()
        self.canvas = tkinter.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack()
        self.btn_snapshot = tkinter.Button(self, text="Start", command=parent.start)
        self.btn_snapshot.pack(anchor='center', side='left')

        self.btn_snapshot = tkinter.Button(self, text="Stop", command=parent.stop)
        self.btn_snapshot.pack(anchor='center', side='left')

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(self, text="Snapshot", command=parent.snapshot)
        self.btn_snapshot.pack(anchor='center', side='left')

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(self, text="Source", command=parent.select_source)
        self.btn_snapshot.pack(anchor='center', side='left')

class ChannelCanvas(tkinter.Frame):
    def __init__(self, parent, text="Meta Encoders", width=None, height=None):
        super().__init__(parent)
        self.width  = width
        self.height = height
        self.label = tkinter.Label(self, text=text)
        self.label.pack()
        self.canvas = tkinter.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack()
        vlist = ['Channel 0', 'Channel 1',
                 'Channel 2', 'Channel 3',
                 'Channel 4', 'Channel 5',
                 'Channel 6', 'Channel 7',
                 'Channel 8']
        self.combo = ttk.Combobox(self, values=vlist)
        self.combo.set('Channel 0')
        self.combo.pack()

    def get_channel_value(self):
        return int(self.combo.get().split()[-1])

class SegmentCanvas(tkinter.Frame):
    def __init__(self, parent, text="Segmentation Mask", width=None, height=None):
        super().__init__(parent)
        self.width  = width
        self.height = height
        self.label = tkinter.Label(self, text=text)
        self.label.pack()
        self.canvas = tkinter.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack()
        # Button that lets the user take a segmentation mask
        self.btn_snapshot = tkinter.Button(self, text="Start", command=parent.start)
        self.btn_snapshot.pack(side='left')
        # Button that lets the user take a segmentation mask
        self.btn_snapshot = tkinter.Button(self, text="Mask", command=parent.stop)
        self.btn_snapshot.pack(side='left')

class SpectralCanvas(tkinter.Frame):
    def __init__(self, parent, text="", width=None, height=None):
        super().__init__(parent)
        self.width  = width
        self.height = height
        self.label = tkinter.Label(self, text=text)
        self.label.pack()
        self.canvas = tkinter.Canvas(self, width=self.width, height=self.height)
        self.canvas.pack()
        self.slider = tkinter.Scale(self, from_=400, to=730, orient='horizontal')
        self.slider.pack(fill='x')
        self.label = tkinter.Label(self, text='wl, nm')
        self.label.pack()

    def get_scale_value(self):
        value = self.slider.get()
        wl = np.linspace(400, 730, 34)
        return np.searchsorted(wl, value)
        #print(self.slider.get())

class tkCamera(tkinter.Frame):

    def __init__(self, parent, camera, text="", source=0, width=None, height=None, sources=None):
        """TODO: add docstring"""

        super().__init__(parent)

        self.source = source
        self.width  = width
        self.height = height
        self.other_sources = sources
        self.segmentation = False
        self.camera = camera

        #self.window.title(window_title)
        self.vid = VideoCapture(self.source, self.width, self.height)

        # self.label = tkinter.Label(self, text=text)
        # self.label.pack()

        self.canvas = RawCanvas(self, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=0, column=0)

        self.canvas_channel = ChannelCanvas(self, width=self.vid.width, height=self.vid.height)
        self.canvas_channel.grid(row=0, column=1)

        self.canvas_spectral = SpectralCanvas(self, text='spectral map', width=self.vid.width, height=self.vid.height)
        self.canvas_spectral.grid(row=1, column=0)

        self.canvas_segment = SegmentCanvas(self, width=self.vid.width, height=self.vid.height)
        self.canvas_segment.grid(row=1, column=1)


        # Button that lets the user take a snapshot
        # self.btn_snapshot = tkinter.Button(self, text="Start", command=self.start)
        # self.btn_snapshot.pack(anchor='center', side='left')
        #
        # self.btn_snapshot = tkinter.Button(self, text="Stop", command=self.stop)
        # self.btn_snapshot.pack(anchor='center', side='left')
        #
        # # Button that lets the user take a snapshot
        # self.btn_snapshot = tkinter.Button(self, text="Snapshot", command=self.snapshot)
        # self.btn_snapshot.pack(anchor='center', side='left')
        #
        # # Button that lets the user take a snapshot
        # self.btn_snapshot = tkinter.Button(self, text="Source", command=self.select_source)
        # self.btn_snapshot.pack(anchor='center', side='left')

        # After it is called once, the update method will be automatically called every delay milliseconds
        # calculate delay using `FPS`
        self.delay = int(1000/self.vid.fps)

        print('[tkCamera] source:', self.source)
        print('[tkCamera] fps:', self.vid.fps, 'delay:', self.delay)

        self.image = None

        self.dialog = None

        self.running = True
        self.update_frame()

    def enable_segment(self):
        self.segmentation = True

    def start(self):
        """TODO: add docstring"""

        #if not self.running:
        #    self.running = True
        #    self.update_frame()
        self.vid.start_recording()

    def stop(self):
        """TODO: add docstring"""

        #if self.running:
        #   self.running = False
        self.vid.stop_recording()

    def snapshot(self):
        """TODO: add docstring"""

        # Get a frame from the video source
        #ret, frame = self.vid.get_frame()
        #if ret:
        #    cv2.imwrite(time.strftime("frame-%d-%m-%Y-%H-%M-%S.jpg"), cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))

        # Save current frame in widget - not get new one from camera - so it can save correct image when it stoped
        #if self.image:
        #    self.image.save(time.strftime("frame-%d-%m-%Y-%H-%M-%S.jpg"))

        self.vid.snapshot()

    def process(self, frame):
        return self.camera.transform(frame)

    def get_photo(self, frame):
        frame = np.uint8(255*frame)
        frame = cv2.resize(frame, (self.width, self.height))
        image = PIL.Image.fromarray(frame)
        return PIL.ImageTk.PhotoImage(image=image)

    def get_spectra(self):
        return self.camera.inverse(self.tensor)

    def update_frame(self):
        """TODO: add docstring"""

        # widgets in tkinter already have method `update()` so I have to use different name -

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            frame = frame / 255.
            self.tensor = self.process(frame)
            self.photo = self.get_photo(frame[:,::-1])
            self.canvas.canvas.create_image(0, 0, image=self.photo, anchor='nw')
            channel_idx = self.canvas_channel.get_channel_value()
            self.photo_channel = self.get_photo(self.tensor[:,:, channel_idx])
            self.canvas_channel.canvas.create_image(0, 0, image=self.photo_channel, anchor='nw')
            self.spectra = self.get_spectra()
            spectra_idx = self.canvas_spectral.get_scale_value()
            self.photo_spectra = self.get_photo(self.spectra[:,:, spectra_idx])
            self.canvas_spectral.canvas.create_image(0, 0, image=self.photo_spectra, anchor='nw')
            self.canvas_segment.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        if self.running:
            self.after(self.delay, self.update_frame)

    def select_source(self):
        """TODO: add docstring"""

        # open only one dialog
        if self.dialog:
            print('[tkCamera] dialog already open')
        else:
            self.dialog = tkSourceSelect(self, self.other_sources)

            #self.label['text'] = self.dialog.name
            self.source = self.dialog.source

            self.vid = VideoCapture(self.source, self.width, self.height)

