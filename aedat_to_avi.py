import sys
import os
import math
import cv2
import numpy as np
import argparse
import utils.dvsproc as dvsproc
import h5py
import glob


def init_database(db_name, save_path):
    """Initialize database for a given dataset.

    Parameters
    ----------
    db_name : string
        Database name
    save_path : string
        the destination of the database

    Returns
    -------
    database : h5py.File
        a HDF5 file object
    """
    # append extension if needed
    if db_name[-5:] != ".hdf5" and db_name[-3:] != ".h5":
        db_name += ".hdf5"

    # create destination folder if needed
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    db_name = os.path.join(save_path, db_name)
    database = h5py.File(db_name, "a")

    return database

def gen_db(database, data_path):
    """Generate UCF50 structure.

    Parameters
    ----------
    database : h5py.File
        HDF5 file object
    ucf50_stats : dictionary
        the dictionary that contains UCF50's stats

    Returns
    -------
    database : h5py.File
        HDF5 file object with multiple groups
    """


    for category in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, category)):
            if category not in database:
                database.create_group(category)
            print("[MESSAGE] Category %s is created" % (category))

    print("[MESSAGE] HDF5 structure is generated.")


def save_recording(record_name, record_data,
                   database, group=None, group_path=None,
                   bounding_box=None, metadata=None, renew=False):
    """Save a given recording to a given group.

    Parameters
    ----------
    record_name : string
        the name of the recording
    record_data : tuple
        a tuple with 4 elements that contains recording data
    database : h5py.File
        a HDF5 file object.
    group : h5py.Group
        a HDF5 group object.
    group_path : string
        The path to the given group from root group.
    bounding_box : numpy.ndarray
        Bounding box information for tracking or detection dataset
    metadata : dictionary
        the metadata that is associated with the recording
    renew : bool
        indicate if the data should be renewed

    Returns
    -------
    A recording saved in the given group
    """
    # Direct to the given group if no Group object presented
    T, X, Y, Pol = record_data

    if group is None:
        if group_path is None:
            raise ValueError("Group path must not be None!")
        group = database[group_path]

    if record_name not in group or renew is True:
        if record_name not in group:
            gp = group.create_group(record_name)
        else:
            gp = group[record_name]

        if renew is True:
            del gp["timestamps"]
            del gp["x_pos"]
            del gp["y_pos"]
            del gp["pol"]

        gp.create_dataset("timestamps", data=T.astype(np.int32),
                          dtype=np.int32)
        gp.create_dataset("x_pos", data=X.astype(np.uint8),
                          dtype=np.uint8)
        gp.create_dataset("y_pos", data=Y.astype(np.uint8),
                          dtype=np.uint8)
        gp.create_dataset("pol", data=Pol.astype(np.bool),
                          dtype=np.bool)



        database.flush()


if __name__ == '__main__':

    # parse the command line argument
    parser = argparse.ArgumentParser(description='create dataset.')
    parser.add_argument('--data_path', 
                    default='../dataset/ActionRecognition',
                    help='The .aedat file dir.')
    # parser.add_argument('--save_path', 
    #                 default='/media/imagr/Data/Projects/AEDAT/dataset/ActionRecognition',
    #                 help='The hdf5 save path.')
    parser.add_argument('--video_path', 
                default='../dataset/ActionRecognitionAVI',
                help='The hdf5 save path.')
    # parser.add_argument('--db_name', 
    #                 default='data',
    #                 help='The hdf5 file name.')
    args = parser.parse_args()

    data_path = args.data_path
    # save_path = args.save_path
    video_path = args.video_path
    # db_name = args.db_name

    num_frames = 36

    # database = init_database(db_name, save_path)
    # gen_db(database, data_path)

    # # Set dataset metadata
    # database.attrs["device"] = 'DAVIS346redColor'
    # # database.attrs["fps"] = 100
    # # database.attrs["monitor_id"] = monitor_id
    # # database.attrs["monitor_feq"] = monitor_feq
    # database.attrs["width"] = 260
    # database.attrs["height"] = 346

    for root, dirnames, filenames in os.walk(data_path):

        for filename in filenames:
            vid_n, vid_ex = os.path.splitext(filename)

            category = os.path.split(root)[-1]

            if vid_ex not in ['.aedat']:
                continue

            record_path = os.path.join(root, filename)
            print("[MESSAGE] Loading %s" % (record_path))

            record_data = dvsproc.loadaerdat(record_path)
            # save_recording(vid_n, record_data, database,
            #                group_path=category)


            T, X, Y, Pol = record_data

            path = os.path.join(video_path, category)

            if not os.path.exists(path):
                os.makedirs(path)

            if (len(T) != 0):

                (T, X, Y, Pol) = dvsproc.clean_up_events(T, X, Y, Pol, window=1000)
                # frames, fs, _ = dvsproc.gen_dvs_frames(T, X, Y, Pol, num_frames, fs=3)
                frames = dvsproc.get_snn_dvs_frames(T, X, Y, Pol, 260, 346, num_frames)

                out = cv2.VideoWriter(os.path.join(path, vid_n + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15, (346, 260))
                 
                # img_count = 1
                for frame in frames:
                    # print(frame)

                    ray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


                    # outputfile = './' + str(img_count) + '.png'
                    # cv2.imwrite(outputfile, frame)

                    # cv2.imshow('frame',ray)
                    # if cv2.waitKey(50) & 0xFF == ord('q'):
                    #     break

                    # img_count += 1

                    out.write(ray)
                out.release()

            else:
                print(os.path.join(path, vid_n + '.avi'))


            print("[MESSAGE] Sequence %s is saved" % vid_n)

    # database.flush()
    # database.close()
    # print("[MESSAGE] dataset is saved to %s" % (save_path))



