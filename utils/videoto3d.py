import numpy as np
import cv2
import os

class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color=False, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # if skip:
        #     frames = [x * nframe / self.depth for x in range(self.depth)]
        # else:
        #     frames = [x for x in range(self.depth)]
        # framearray = []

        # for i in range(self.depth):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        #     ret, frame = cap.read()
        #     frame = cv2.resize(frame, (self.height, self.width))
        #     if color:
        #         framearray.append(frame)
        #     else:
        #         framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


        framearray = []
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:

                if color:
                    framearray.append(frame)
                else:
                    frame = cv2.resize(frame, (self.height, self.width))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


                    # cv2.imshow('img',frame)
   
                    # cv2.waitKey(5)

                    framearray.append(frame)
            else:
                break       

        cap.release()

        return np.array(framearray)

        # framearray = []

        # for file in os.listdir(filename):
        #     frame = cv2.imread(os.path.join(filename, file),cv2.IMREAD_GRAYSCALE)

        #     frame = cv2.resize(frame, (self.height, self.width))

        #     framearray.append(frame)


        # return np.array(framearray)


    def get_classname(self, filename):
        # return filename[filename.find('_') + 1:filename.find('_', 2)]

        return filename.split('/')[-1]

