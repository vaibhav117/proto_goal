import os
import sys

import imageio
import numpy as np

import utils


class VideoRecorder(object):
    def __init__(self, root_dir, height=500, width=500, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render("rgb_array", width=self.width, height=self.height)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


## TODO render better quality videos with the marker 
## TODO remove the red dot from the observation value
