import struct
import os
import numpy as np
import cv2

def getDVSeventsDavis(file, numEvents=1e10, startTime=0):
    """ DESCRIPTION: This function reads a given aedat file and converts it into four lists indicating 
                     timestamps, x-coordinates, y-coordinates and polarities of the event stream. 
    
    Args:
        file: the path of the file to be read, including extension (str).
        numEvents: the maximum number of events allowed to be read (int, default value=1e10).
        startTime: the start event timestamp (in microseconds) where the conversion process begins (int, default value=0).

    Return:
        ts: list of timestamps in microseconds.
        x: list of x-coordinates in pixels.
        y: list of y-coordinates in pixels.
        pol: list of polarities (0: on -> off, 1: off -> on).       
    """
    print('\ngetDVSeventsDavis function called \n')
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY
   

    print('Reading in at most', str(numEvents))
    

    triggerevent = int('400', 16)
    polmask = int('800', 16)
    xmask = int('003FF000', 16)
    ymask = int('7FC00000', 16)
    typemask = int('80000000', 16)
    typedvs = int('00', 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []
    pol = []
    numeventsread = 0
    
    length = 0
    aerdatafh = open(file, 'rb')
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from('>II', tmp)
        ad = abs(ad)
        if tm >= startTime:
            if (ad & typemask) == typedvs:
                xo = sizeX - 1 - float((ad & xmask) >> xshift)
                yo = float((ad & ymask) >> yshift)
                polo = 1 - float((ad & polmask) >> polshift)
                if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                    x.append(xo)
                    y.append(yo)
                    pol.append(polo)
                    ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    print('Total number of events read =', numeventsread)
    print('Total number of DVS events returned =', len(ts))

    return ts, x, y, pol


def get_sae_dvs_frames(T, X, Y, Pol, width, height):
    step_time = 10000 #The cumulative time of a frame

    start_idx = 0
    end_idx = 0
    start_time = T[0]
    end_time = start_time + step_time

    img_count = 0

    img_array = []

    while end_time <= T[-1]: 
        while T[end_idx] < end_time:
            end_idx = end_idx + 1
        
        data_x = np.array(X[start_idx:end_idx]).reshape((-1, 1))
        data_y = np.array(Y[start_idx:end_idx]).reshape((-1, 1))
        data_T = np.array(T[start_idx:end_idx]).reshape((-1, 1))
        data = np.column_stack((data_x, data_y)).astype(np.int32)
        
        timestamp = start_time*np.ones((width, height))
        
        for i in range(0, data.shape[0]):
            timestamp[data[i,1], data[i,0]]=data_T[i]
           
        grayscale = np.flip(255*(timestamp-start_time)/step_time, 0).astype(np.uint8)#The normalization formula
        vis2 = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
        img_array.append(vis2)

        start_time = end_time
        end_time += step_time
        start_idx = end_idx
        img_count += 1

    return img_array



def gen_dvs_frames(timestamps, xaddr, yaddr, pol, num_frames, fs=3,
                   platform="linux2", device="DAVIS240"):
    """Generate DVS frames from recording.

    Paramters
    ---------
    timestamps : numpy.ndarray
        time stamps record
    xaddr : numpy.ndarray
        x position of event recordings
    yaddr : numpy.ndarry
        y position of event recordings
    pol : nujmpy.ndarray
        polarity of event recordings
    num_frames : int
        number of frames in original video sequence
    fs : int
        maximum of events of a pixel
    platform : string
        recording platform of the source. Available option:
        "macosx", "linux2"
    device : string
        DVS camera model - "DAVIS240" (default), "DVS128", "ATIS"

    Returns
    -------
    frames : list
        list of DVS frames
    fs : int
        a scale factor for displaying the frame
    ts : list
        a list that records start timestamp for each frame
    """
    base = 0
    max_events_idx = timestamps.shape[0]-1
    time_step = (timestamps[-1]-timestamps[0])/num_frames
    if device == "DAVIS240":
        base_frame = np.zeros((180, 240), dtype=np.int8)
    elif device == "DVS128":
        base_frame = np.zeros((128, 128), dtype=np.int8)
    elif device == "ATIS":
        base_frame = np.zeros((240, 304), dtype=np.int8)
    else:
        base_frame = np.zeros((180, 240), dtype=np.int8)

    print("Average frame time: %i" % (time_step))

    frames = []
    ts = []
    while base < max_events_idx and len(frames) < num_frames:
        ts.append(timestamps[base])
        k = base
        diff = 0
        frame = base_frame.copy()
        while diff < time_step and k < max_events_idx:
            if platform == "linux2":
                if device == "DAVIS240":
                    x_pos = min(239, xaddr[k]-1)
                elif device == "DVS128":
                    x_pos = min(127, xaddr[k]-1)
                elif device == "ATIS":
                    x_pos = min(304, xaddr[k]-1)
            elif platform == "macosx":
                if device == "DAVIS240":
                    x_pos = min(239, 240-xaddr[k])
                elif device == "DVS128":
                    x_pos = min(127, 128-xaddr[k])
            if device == "DAVIS240":
                y_pos = min(179, 180-yaddr[k])
            elif device == "DVS128":
                y_pos = min(127, yaddr[k])
            elif device == "ATIS":
                y_pos = min(240, yaddr[k])

            if pol[k] == 1:
                frame[y_pos, x_pos] = min(fs, frame[y_pos, x_pos]+1)
            elif pol[k] == 0:
                frame[y_pos, x_pos] = max(-fs, frame[y_pos, x_pos]-1)
            k += 1
            diff = int(timestamps[k]-timestamps[base])

        base = k-1
        frames.append(frame)

    return frames, fs, ts
