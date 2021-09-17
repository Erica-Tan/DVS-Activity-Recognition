"""DVS imaging related functions.
This module is written under extreamly helpful codes under:
https://sourceforge.net/p/jaer/code/HEAD/tree/scripts/python/
https://github.com/duguyue100/spikefuel
https://github.com/CrystalMiaoshu/PAFBenchmark/
"""
import math
import struct
import os

import cv2
import numpy as np

from utils.snn import SNN

V3 = "aedat3"
V2 = "aedat"  # current 32bit file format
V1 = "dat"  # old format

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def check_aedat(datafile):
    """Check Common AEDAT errors.

    Paramters
    ---------
    datafile : string
        path to the datafile

    Returns
    -------
    flag : bool
        boolean flag that indicates the error of the file
    """
    aerdatafh = open(datafile, 'rb')
    fileinfo = os.stat(datafile)

    flag = True
    # check if empty events
    if (fileinfo.st_size <= 131731):
        # means 0 event wrote or unsuccessful writing
        print("[MESSAGE] FILE TOO SHORT")
        flag = False

    # check if first byte of data is #
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        if str(lt)[:4] == "#End":
            lt = aerdatafh.readline()
            if not str(lt):
                print("[MESSAGE] FILE NO EVENT WRITTEN")
                flag = False
            elif lt[0] == "#":
                flag = False
            return fileinfo.st_size, flag

        lt = aerdatafh.readline()

    return fileinfo.st_size, flag


def loadaerdat(datafile, length=0, version=V2, debug=0, camera='DAVIS346redColor'):
    """Load AER data file and parse these properties of AE events.

    Source:
    https://sourceforge.net/p/jaer/code/HEAD/tree/scripts/python/jAER_utils/loadaerdat.py

    Paramters
    ---------
    datafile : string
        absolute path to the file to read
    length : int
        how many bytes(B) should be read; default 0=whole file
    version : string
        which file format version is used: "aedat" = v2, "dat" = v1 (old)
    debug : int
        0 = silent, 1 (default) = print summary, >=2 = print all debug
    camera : string
        'DVS128' or 'DAVIS240'

    Returns
    -------
    timestamps : int
        time tamps (in us)
    xaddr : int
        x posititon
    yaddr : int
        y position
    pol : int
        polarity
    """
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us
    if(camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif(camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003ff000
        xshift = 12
        ymask = 0x7fc00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    elif(camera == 'DAVIS346redColor'):
        pmask = 0x800
        xmask = 0x003ff000
        ymask = 0x7fc00000
        xshift = 12
        yshift = 22
        pshift = 11
        eventtypeshift = int('80000000', 16)
        
    else:
        raise ValueError("Unsupported camera: %s" % (camera))

    if (version == V1):
        print ("using the old .dat format")
        aeLen = 6
        readMode = '>HI'  # ushot, ulong = 2B+4B

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size
    if debug > 0:
        print ("file size", length)



    if(camera == 'DAVIS346redColor'):
        lt = aerdatafh.readline()
        while lt and str(lt)[2] == "#":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline()
            continue
    else:  
        # header
        lt = aerdatafh.readline()
        while lt and lt[0] == "#":
            #  or str(lt)[:4] == "# cr"
            if str(lt)[:4] == "#End":
                p += len(lt)
                k += 1
                lt = aerdatafh.readline()
                break
            p += len(lt)
            k += 1
            lt = aerdatafh.readline()
            if debug >= 2:
                print (str(lt))
            # continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    if debug > 0:
        print (xmask, xshift, ymask, yshift, pmask, pshift)

    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if(camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS

        # parse event's data
        if(eventtype == EVT_DVS):  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift

            if debug >= 3:
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    if debug > 0:
        try:
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (
                   len(timestamps), len(timestamps) / float(10 ** 6),
                   (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (
                   timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print ("failed to print statistics")

    timestamps = np.array(timestamps)
    xaddr = np.array(xaddr)
    yaddr = np.array(yaddr)
    pol = np.array(pol)
    return timestamps, xaddr, yaddr, pol


def find_nearest(array, value):
    """Find nearest value in an array.

    Parameters
    ----------
    array : numpy.ndarray
        1-d array
    value : int
        the given value
    """
    return (np.abs(array-value)).argmin()


def cal_event_freq(event_arr, window=1000):
    """Calculate event frequence by given time window.

    Parameters
    ----------
    event_arr : numpy.ndarray
        array has 2 rows, first row contains timestamps,
        second row consists of corresponding event count at particular
        timestep
    window : int
        sliding window over timestamps, by default, it's 1000 us = 1ms

    Returns
    -------
    event_freq : numpy.ndarray
        Event frequency count in given window
    """
    idx = 0
    tot_idx = event_arr.shape[1]-1
    event_freq = []
    while idx < tot_idx:
        end_idx = find_nearest(event_arr[0, idx:min(idx+window, tot_idx)],
                               event_arr[0, idx]+window)
        end_idx = end_idx+idx+1

        event_freq.append(np.array([event_arr[0, end_idx],
                                    np.sum(event_arr[1, idx:end_idx])]))
        idx = end_idx+1

    return np.array(event_freq)


def cal_event_count(timestamps):
    """Calculate event count based on timestamps.

    Parameters
    ----------
    timestamps : numpy.ndarray
        timestamps array in 1D array

    Returns
    -------
    event_arr : numpy.ndarray
        array has 2 rows, first row contains timestamps,
        second row consists of corresponding event count at particular
        timestep
    """
    event_ts, event_count = np.unique(timestamps, return_counts=True)

    return np.asarray((event_ts, event_count))


def cal_running_std(event_freq, n=16):
    """Calculate running standard deviation.

    Parameters
    ----------
    event_freq : numpy.ndarray
        Event frequency count in given window
    n : int
        Running window for computing STD

    Returns
    -------
    o : numpy.ndarray
        Running standard deviation of given event frequency array
    """
    q = event_freq[:, 1]**2
    q = np.convolve(q, np.ones((n, )), mode="valid")
    s = np.convolve(event_freq[:, 1], np.ones((n, )), mode="valid")
    o = (q-s**2/n)/float(n-1)

    return o


def cal_first_response(timestamps, window=1000):
    """Calculate the first event burst based on standard deviation.

    Parameters
    ----------
    timestamps : numpy.ndarray
        time stamps record
    window : int
        sliding window over timestamps, by default, it's 1000 us = 1ms

    Returns
    -------
    key_idx : int
        The start index of the first event burst
    """
    # Calculate event count
    event_info = cal_event_count(timestamps)


    # calculate events number within peroid of agiven window
    event_freq = cal_event_freq(event_info, window=window)

    # print(event_freq.shape)

    # calculate running standard deviation
    n = 8
    o = cal_running_std(event_freq, n)

    # print(o.shape)

    start_idx = 0
    while (o[start_idx+1]/o[start_idx] < 3):
        start_idx += 1

        if start_idx+1 > len(o)-1:
            break

    key_ts = event_freq[start_idx+2*n, 0]

    return np.where(timestamps == key_ts)[0][0]


def clean_up_events(timestamps, xaddr, yaddr, pol, window=1000,
                    key_idx=0):
    """Clean up event series based on standard deviation.

    Parameters
    ----------
    timestamps : numpy.ndarray
        time stamps record
    xaddr : numpy.ndarray
        x position of event recordings
    yaddr : numpy.ndarry
        y position of event recordings
    pol : nujmpy.ndarray
        polarity of event recordings
    window : int
        sliding window over timestamps, by default, it's 1000 us = 1ms
    key_idx : int
        the timestamp that indicates the first event burst

    Returns
    -------
    Cleaned signal
    """
    if key_idx == 0:
        key_idx = cal_first_response(timestamps, window)

    return (timestamps[key_idx:], xaddr[key_idx:], yaddr[key_idx:],
            pol[key_idx:])

    # # Calculate event count
    # event_info = cal_event_count(timestamps)
    #
    # # calculate events number within peroid of agiven window
    # event_freq = cal_event_freq(event_info, window=window)
    #
    # # calculate running standard deviation
    # n = 16
    # o = cal_running_std(event_freq, n)
    #
    # start_idx = 0
    # while (o[start_idx+1]/o[start_idx] < 3):
    #     start_idx += 1
    #
    # key_ts = event_freq[start_idx+2*n, 0]
    #
    # key_idx = np.where(timestamps == key_ts)[0][0]
    #
    # return (timestamps[key_idx:], xaddr[key_idx:], yaddr[key_idx:],
    #         pol[key_idx:])


def remove_outliers(idx_ts):
    """Remove outliers from a given timestamps array.

    Parameters
    ----------
    idx_ts : numpy.ndarray
        given timestamp array.

    Returns
    -------
    idx_ts_new : numpy.ndarray.
        outliers removed timestamp array.
    """
    idx_ts_new = idx_ts.copy()
    summary = np.percentile(idx_ts_new, [25, 50, 75])
    high_lim = summary[0] - 1.5 * (summary[2] - summary[1])
    low_lim = summary[2] + 1.5 * (summary[2] - summary[1])

    idx_ts_new = idx_ts_new[~(idx_ts_new >= low_lim)]
    idx_ts_new = idx_ts_new[~(idx_ts_new <= high_lim)]

    return idx_ts_new


def gen_dvs_frames(timestamps, xaddr, yaddr, pol, num_frames, fs=3,
                   platform="linux2", device="DAVIS346redColor"):
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
        base_frame = np.zeros((260, 346), dtype=np.int8)

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
                else:
                    x_pos = min(345, xaddr[k]-1)
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
            else:
                y_pos = min(259, 260-yaddr[k])

            if pol[k] == 1:
                # print(min(fs, frame[y_pos, x_pos]+1))
                frame[y_pos, x_pos] = min(fs, frame[y_pos, x_pos]+1)
            elif pol[k] == 0:
                frame[y_pos, x_pos] = max(-fs, frame[y_pos, x_pos]-1)
            k += 1
            diff = int(timestamps[k]-timestamps[base])

        base = k-1
        frames.append((((frame+fs)/float(2*fs))*255).astype(np.uint8))

    return frames, fs, ts


def get_sae_dvs_frames(T, X, Y, Pol, width, height, num_frames):
    """
    SAE(Surface of Active Events) encoding method
    source https://github.com/CrystalMiaoshu/PAFBenchmark/blob/master/sae.py
    """
    T = T.reshape((-1, 1))
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    Pol = Pol.reshape((-1, 1))

    # step_time = 10000 #The cumulative time of a frame

    step_time = math.ceil((T[-1]-T[0])/(num_frames+1))

    start_idx = 0
    end_idx = 0
    start_time = T[0]
    end_time = start_time + step_time

    frames = []

    while end_time <= T[-1] and len(frames) < num_frames:
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
        frames.append(grayscale)

        start_time = end_time
        end_time += step_time
        start_idx = end_idx

    return frames


def get_snn_dvs_frames(T, X, Y, Pol, width, height, num_frames):
    """
    LIF(Leaky Integrate-and-Fire) encoding method
    source https://github.com/CrystalMiaoshu/PAFBenchmark/blob/master/snn.py
    """
    T = np.array(T).reshape((-1, 1))
    X = np.array(X).reshape((-1, 1))
    Y = np.array(Y).reshape((-1, 1))
    Pol = np.array(Pol).reshape((-1, 1))
    data = np.hstack((T, X, Y, Pol))

    step_time = math.ceil((T[-1]-T[0])/(num_frames+1))

    dvs_snn = SNN(width, height, step_time)
    dvs_snn.spiking(data)

    # print(step_time, len(dvs_snn.frames))

    return dvs_snn.frames


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
