import numpy as np
from os import listdir
from os.path import join
import cv2

class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color=False, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        framearray = []
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:

                if color:
                    framearray.append(frame)
                else:
                    frame = cv2.resize(frame, (self.height, self.width))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    framearray.append(frame)
            else:
                break       

        cap.release()

        return np.array(framearray)

    def get_classname(self, filename):
        return filename.split('/')[-1]

def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift

    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events


class PAFBDataset:
    def __init__(self, root, augmentation=False):
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            self.files += new_files
            self.labels += [i] * len(new_files)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """

        label = self.labels[idx]
        f = self.files[idx]

        events = np.load(f).astype(np.float32)/float(255)

        return events, label