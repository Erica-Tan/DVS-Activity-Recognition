import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import videoto3d
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def loaddata(video_dir, vid3d, nclass, result_dir):
    files = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []

    pbar = tqdm(total=len(files))

    for label in files:
        for video in os.listdir(os.path.join(video_dir, label)):
            pbar.update(1)

            name = os.path.join(video_dir, label, video)

            

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
            

    pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num

    return np.array(X).transpose((0, 2, 3, 1)), labels



def train_model(train_data,train_labels,validation_data,validation_labels):
  ''' used fully connected layers, SGD optimizer and 
      checkpoint to store the best weights'''

  model = Sequential()
  model.add(Flatten(input_shape=train_data.shape[1:]))
  model.add(Dense(512, activation='relu'))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(5, activation='softmax'))
  sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  model.load_weights('video_3_512_VGG_no_drop.h5')
  callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_3_512_VGG_no_drop.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
  nb_epoch = 500
  model.fit(train_data,train_labels,validation_data = (validation_data,validation_labels),batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)
  return model

def main():
    parser = argparse.ArgumentParser(description='2D convolution')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--videos', type=str, default='/media/imagr/Data/AEDAT/dataset/ActionRecognitionAVI2',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=10)
    parser.add_argument('--output', type=str, default='./')
    args = parser.parse_args()

    img_rows, img_cols, frames = 128, 128, 36

    # fname_npz = os.path.join(args.videos, 'dataset_10_36_True.npz')


    nb_classes = args.nclass
    # if os.path.exists(fname_npz):
    #     loadeddata = np.load(fname_npz)
    #     X, Y = loadeddata["X"], loadeddata["Y"]
    #     X = X.reshape((X.shape[0], img_rows, img_cols, frames))
    # else:
    vid3d = videoto3d.Videoto3D(img_rows, img_cols, frames)

    x, y = loaddata(args.videos, vid3d, args.nclass, args.output)
    X = x.reshape((x.shape[0], img_rows, img_cols, frames))
    Y = np_utils.to_categorical(y, nb_classes)

    X = X.astype('float32')
    # np.savez(fname_npz, X=X, Y=Y)
    # print('Saved dataset to dataset.npz.')
    print('X_shape:{}\nY_shape:{}'.format(X.shape, Y.shape))

    print('input shape:{}'.format(X.shape[1:]))


    # # exit()


    # # define model
    # model = Sequential()

    # model.add(Convolution2D(32, 3, 3, border_mode='same',
    #                         input_shape=X.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(32, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Convolution2D(64, 3, 3, border_mode='same'))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, 3, 3))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    # model.summary()
    # # plot_model(model, show_shapes=True, to_file=os.path.join(args.output, 'model.png'))

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)


    # # Set up TensorBoard
    # tensorboard = TensorBoard(batch_size=args.batch)
    # scheduler = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=1e-5)

    # filepath="./checkpoint/weights-{epoch:02d}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [tensorboard, scheduler, checkpoint]

    # history = model.fit(X_train, y_train,
    #                     batch_size=args.batch,
    #                     nb_epoch=args.epoch,
    #                     validation_data=(X_val, y_val),
    #                     shuffle=True, callbacks=callbacks_list)
    # # model_json = model.to_json()
    # # with open(os.path.join(args.output, 'ucf101cnnmodel.json'), 'w') as json_file:
    # #     json_file.write(model_json)
    # # model.save_weights(os.path.join(args.output, 'ucf101cnnmodel.hd5'))

    # # loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    # # print('Test loss:', loss)
    # # print('Test accuracy:', acc)
    # # plot_history(history, args.output)
    # # save_history(history, args.output)

if __name__ == '__main__':
    main()
