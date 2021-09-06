import os
import numpy as np
import cv2

from torch.utils.data import DataLoader, random_split

import utils.dvsproc as dvsproc


class PAFBDataset:
    def __init__(self, root, augmentation=False):
        self.classes = os.listdir(root)
        self.img_rows, self.img_cols, self.num_frames = 128, 128, 36

        self.events = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            for f in os.listdir(os.path.join(root, c)):
                print(os.path.join(root, c, f))

                record_data = dvsproc.loadaerdat(os.path.join(root, c, f))
                T, X, Y, Pol = record_data

                framearray = []
                if (len(T) != 0):
                    (T, X, Y, Pol) = dvsproc.clean_up_events(T, X, Y, Pol, window=1000)
                    frames, fs, _ = dvsproc.gen_dvs_frames(T, X, Y, Pol, self.num_frames, fs=3)
                    # frames = dvsproc.get_snn_dvs_frames(T, X, Y, Pol, 260, 346, self.num_frames)

                    # i = 0
                    for frame in frames:
                        # print(type(frame), frame.shape)
                        frame = cv2.resize(frame, (self.img_rows, self.img_cols))
                        # cv2.imwrite(os.path.join("./", "%d-%d.png" % (label, i)), frame)
                        framearray.append(frame)

                        # i += 1

                    self.events.append(np.array(framearray))
                    self.labels.append(i)



    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        event = self.events[idx]

        return event.astype(np.float32)/float(255), label

def test():
    data = PAFBDataset('./dataset/test/')

    test_size = int(0.2 * len(data))
    train_size = len(data) - test_size
    train_ds, test_ds = random_split(data, [train_size, test_size])

    loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True
    )

    for batch_idx, (x, y_true) in enumerate(loader):
        print(x.shape, y_true.shape)

        break

if __name__ == "__main__":
    test()

