import os
import numpy as np
import cv2

def video3d(filename, width, height, color=False):
    cap = cv2.VideoCapture(filename)
    # nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    framearray = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:

            if color:
                framearray.append(frame)
            else:
                frame = cv2.resize(frame, (height, width))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                framearray.append(frame)
        else:
            break

    cap.release()

    return np.array(framearray)

class PAFBDataset:
    def __init__(self, root, augmentation=False):
        self.classes = os.listdir(root)
        self.img_rows, self.img_cols, self.frames = 128, 128, 36

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            new_files = [os.path.join(root, c, f) for f in os.listdir(os.path.join(root, c))]
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