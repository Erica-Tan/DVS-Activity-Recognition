"""This python file is a way to process raw data through LIF
source https://github.com/CrystalMiaoshu/PAFBenchmark/blob/master/snn.py
"""
import cv2
import numpy as np


class SNN():
    """Spiking Neural Network.
    ts: timestamp list of the event stream.
    x: x-coordinate list of the event stream.
    y: y-coordinate list of the event stream.
    pol: polarity list of the event stream.
    threshold: threshold of neuron firing.
    decay: decay of MP with time.
    margin: margin for lateral inhibition.
    spikeVal: MP increment for each event.
    network: MP of each neuron.
    timenet: firing timestamp for each neuron.
    firing: firing numbers for each neuron.
    image: converted output grayscale image.
    """

    def __init__(self, width, height, step_time):
        self.ts = []
        self.x = []
        self.y = []
        self.pol = []
        self.threshold = 1.2
        self.decay = 0.02
        self.margin = 3
        self.spikeVal = 1
        self.network = np.zeros((width, height), dtype=np.float64)
        self.timenet = np.zeros((width, height), dtype=np.int64)
        self.firing = np.zeros((width, height), dtype=np.int64)
        self.image = np.zeros((width, height), dtype=np.int64)
        self.step_time = step_time
        self.frames = []

    def init_timenet(self, t):
        """initialize the timenet with timestamp of the first event"""
        self.timenet[:] = t

    def spiking(self, data):
        """"main process"""
        count = 0
        img_count = 0
        startindex = 0

        for line in data:
            # print(line)
            self.ts.insert(count, int(line[0]))
            self.x.insert(count, int(line[1]))
            self.y.insert(count, int(line[2]))
            self.pol.insert(count, int(line[3]))

            if count == 0:
                self.init_timenet(self.ts[0])
                starttime = self.ts[0]

            self.neuron_update(count, self.spikeVal)

            if self.ts[count] - starttime > self.step_time:
                # print(img_count, count)
                self.show_image(img_count)
                img_count += 1
                starttime = self.ts[count]
                self.image *= 0
                self.firing *= 0

            count += 1

        # print(self.frames)
        print('done')

    def clear_neuron(self, position):
        """reset MP value of the fired neuron"""
        for i in range((-1) * self.margin, self.margin):
            for j in range((-1) * self.margin, self.margin):
                if position[0] + i < 0 or position[0] + i >= 180 or position[1] + j < 0 or position[1] + j >= 180:
                    continue
                else:
                    self.network[position[0] + i][position[1] + j] = 0.0

    def neuron_update(self, i, spike_value):
        """update the MP values in the network"""
        x = self.x[i]
        y = self.y[i]
        escape_time = (self.ts[i] - self.timenet[y][x]) / 1000.0
        residual = max(self.network[y][x] - self.decay * escape_time, 0)
        self.network[y][x] = residual + spike_value
        self.timenet[y][x] = self.ts[i]
        if self.network[y][x] > self.threshold:
            self.firing[y][x] += 1  # countor + 1
            self.clear_neuron([x, y])

    def show_image(self, img_count):
        """convert to and save grayscale images"""
        self.image = np.flip(255 * 2 * (1 / (1 + np.exp(-self.firing)) - 0.5), 0).astype('uint8')
        frame = np.copy(self.image)
        self.frames.append(frame)

        # outputfile = './' + str(img_count) + '.png'
        # cv2.imwrite(outputfile, self.image)

        # outputfile = '/home/imagr/PAFBenchmark/dataset/' + str(img_count) + '.png'
        # cv2.imshow('img', self.image)
        # cv2.waitKey(5)
        # cv2.imwrite(outputfile, self.image)
