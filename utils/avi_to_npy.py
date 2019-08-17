import os
import h5py
import cv2
import numpy as np
import dvsproc

import glob
from videoto3d import Videoto3D
import tqdm
from sklearn.model_selection import train_test_split

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
                frames, fs, _ = dvsproc.gen_dvs_frames(T, X, Y, Pol, num_frames, fs=3)

                # frames = dvsproc.get_sae_dvs_frames(T, X, Y, Pol, 260, 346, num_frames)
                
                # i = 0
                # for frame in frames:
                #     img_name = os.path.join(path, "%s_%d.png" % (name, i))
                #     cv2.imwrite(img_name, frame)

                #     i += 1


                out = cv2.VideoWriter(os.path.join(path, name + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 15, (346, 260))
                 
                for frame in frames:
                    ray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    out.write(ray)
                out.release()

            else:
                print(os.path.join(path, name + '.avi'))

            break

        break



def check_video(video):
    cap = cv2.VideoCapture(video)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print(length, fps)

    # i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(os.path.join('/media/imagr/Data/AEDAT/dataset/ActionRecognitionAVI1', "%d.png" % (i)), gray)
            cv2.imshow('frame',gray)

            # i += 1

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()   


def loaddata(video_dir, vid3d, nclass=10, result_dir='./'):
    files = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []

    # pbar = tqdm(total=len(files))

    for label in files:
        for video in os.listdir(os.path.join(video_dir, label)):
            # pbar.update(1)

            name = os.path.join(video_dir, label, video)

            # print(name)

            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)

            framearray = vid3d.video3d(name)

            if len(framearray) != 0:
                X.append(framearray)

                labels.append(label)

            else:
                print(name)
            

    # pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num

    return np.array(X), labels


def show(img, count):
    npimg = np.array(img)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    plt.imsave('image_new_' + str(count) +'.jpg',np.transpose(npimg, (1,2,0)) )


if __name__ == '__main__':

    # db_path = "/media/imagr/Data/AEDAT/dataset/ActionRecognition/data.hdf5"
    video_path = "/media/imagr/Data/AEDAT/dataset/ActionRecognitionAVI2"
    save_path = "/media/imagr/Data/AEDAT/dataset/ActionRecognitionAVI2Npy"
    # num_frames = 36


    # hdf5_to_video(db_path, video_path, num_frames)

    # video = '/media/imagr/Data/AEDAT/dataset/ActionRecognitionAVI/waving/chenjieneng_10.1.avi'
    # check_video(video)

    img_rows, img_cols, frames = 128, 128, 36

    vid3d = Videoto3D(img_rows, img_cols, frames)
    # x, y = loaddata(video_path, vid3d)

    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


    # np.save('test', X_train[0]) 

    # print(X_train[0].shape)




    # labellist = ["armcrossing" ,"getup" ,"jumping" ,"kicking" ,"pickingup" ,"sitdown" ,"throwing" ,"turningaround" ,"walking" ,"waving"]

    # for idx in range(X_train.shape[0]):
    #     path = os.path.join(save_path, "training", labellist[y_train[idx]])
    #     if not os.path.isdir(path):
    #         os.mkdir(path)

    #     np.save(os.path.join(path, labellist[y_train[idx]] +  "_" + str(idx)), X_train[idx]) 

    # for idx in range(X_val.shape[0]):
    #     path = os.path.join(save_path, "validation", labellist[y_val[idx]])
    #     if not os.path.isdir(path):
    #         os.mkdir(path)

    #     np.save(os.path.join(path, labellist[y_val[idx]] +  "_" + str(idx)), X_val[idx]) 

    # for idx in range(X_test.shape[0]):
    #     path = os.path.join(save_path, "testing", labellist[y_test[idx]])
    #     if not os.path.isdir(path):
    #         os.mkdir(path)

    #     np.save(os.path.join(path, labellist[y_test[idx]] +  "_" + str(idx)), X_test[idx]) 


    # events = np.array(vid3d.video3d('/media/imagr/Data/AEDAT/dataset/ActionRecognitionAVI2/arm_crossing/liuxingbo_1.2.avi', color=False, skip=True))

    events = np.load('/media/imagr/Data/AEDAT/dataset/N-Caltech101/testing/accordion/accordion_0.npy').astype(np.float32)

# 
    # for idx in range(events.shape[0]):
    #     cv2.imshow('frame',events[idx])
    #     if cv2.waitKey(100) & 0xFF == ord('q'):
    #         break