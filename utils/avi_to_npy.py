import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.dvsproc import video3d, gen_dvs_frames

labellist = ["armcrossing", "getup", "jumping", "kicking", "pickingup", "sitdown", "throwing", "turningaround",
             "walking", "waving"]


def hdf5_to_video(db_path, video_path, num_frames):
    db = h5py.File(db_path, mode="r")

    for rec in list(db.keys()):
        print(rec)

        path = os.path.join(video_path, rec)

        if not os.path.exists(path):
            os.makedirs(path)

        for name in list(db[rec].keys()):
            T = db[rec][name]["timestamps"][()]
            X = db[rec][name]["x_pos"][()]
            Y = db[rec][name]["y_pos"][()]
            Pol = db[rec][name]["pol"][()]
            print("[MESSAGE] Reading " + name)

            if (len(T) != 0):

                # (T, X, Y, Pol) = dvsproc.clean_up_events(T, X, Y, Pol, window=1000)
                frames, fs, _ = gen_dvs_frames(T, X, Y, Pol, num_frames, fs=3)

                # frames = dvsproc.get_sae_dvs_frames(T, X, Y, Pol, 260, 346, num_frames)

                # i = 0
                # for frame in frames:
                #     img_name = os.path.join(path, "%s_%d.png" % (name, i))
                #     cv2.imwrite(img_name, frame)

                #     i += 1

                out = cv2.VideoWriter(os.path.join(path, name + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15,
                                      (346, 260))

                for frame in frames:
                    ray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(ray)
                out.release()

            else:
                print(os.path.join(path, name + '.avi'))

            break

        break


def check_video(video, path):
    cap = cv2.VideoCapture(video)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(length, fps)

    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(path, "%d.png" % (i)), gray)
            # cv2.imshow('frame',gray)

            i += 1

            # if cv2.waitKey(30) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def loaddata(video_dir, nclass, img_rows, img_cols, frames):
    files = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []

    pbar = tqdm(total=len(files))

    for label in files:
        for video in os.listdir(os.path.join(video_dir, label)):
            pbar.update(1)

            name = os.path.join(video_dir, label, video)

            # print(name)

            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)

            framearray = video3d(name, img_rows, img_cols)

            if len(framearray) != 0:
                X.append(framearray)

                labels.append(label)

            else:
                print(name)

    pbar.close()

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num

    return np.array(X), labels


def show(img, count):
    npimg = np.array(img)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

    plt.imsave('image_new_' + str(count) + '.jpg', np.transpose(npimg, (1, 2, 0)))


def save_data(X, y, save_path, type):
    save_dir = os.path.join(save_path, type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(X.shape[0]):
        path = os.path.join(save_dir, labellist[y[idx]])
        if not os.path.isdir(path):
            os.mkdir(path)

        print(os.path.join(path, labellist[y[idx]] + "_" + str(idx)))
        np.save(os.path.join(path, labellist[y[idx]] + "_" + str(idx)), X_train[idx])


if __name__ == '__main__':
    video_path = "./dataset/ActionRecognitionAVI"
    save_path = "./dataset/ActionRecognitionAVINpy"

    img_rows, img_cols, frames = 128, 128, 36

    x, y = loaddata(video_path, 10, img_rows, img_cols, frames)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    save_data(X_train, y_train, save_path, "training")
    save_data(X_val, y_val, save_path, "validation")
    save_data(X_test, y_test, save_path, "testing")
