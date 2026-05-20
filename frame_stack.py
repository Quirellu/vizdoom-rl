from collections import deque
import numpy as np

class FrameStack:

    def __init__(self, stack_size):

        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):

        self.frames.clear()

        for _ in range(self.frames.maxlen):
            self.frames.append(frame)

        return np.stack(self.frames, axis=0)

    def step(self, frame):
        frames = np.array(self.frames)

        frames = np.transpose(
            frames,
            (0, 3, 1, 2)
        )

        c = frames.shape[0] * frames.shape[1]

        h = frames.shape[2]

        w = frames.shape[3]

        frames = frames.reshape(c, h, w)

        return frames