import sys
import os
import math
import cv2
import numpy as np
import argparse
import dvsproc

 # parse the command line argument
parser = argparse.ArgumentParser(description='SAE for encoding.')
parser.add_argument('file_path', 
                help='The .aedat file path.')
args = parser.parse_args()

inputfile = args.file_path

T, X, Y, Pol = dvsproc.loadaerdat(inputfile, debug=1)#Read the quaternion array
T = T.reshape((-1, 1))
X = X.reshape((-1, 1))
Y = Y.reshape((-1, 1))
Pol = Pol.reshape((-1, 1))

step_time = 20000 #The cumulative time of a frame

start_idx = 0
end_idx = 0
start_time = T[0]
end_time = start_time + step_time
print(start_time, end_time, T[-1])

img_count = 0

while end_time <= T[-1]: 
    while T[end_idx] < end_time:
        end_idx = end_idx + 1
    
    data_x = np.array(X[start_idx:end_idx]).reshape((-1, 1))
    data_y = np.array(Y[start_idx:end_idx]).reshape((-1, 1))
    data_T = np.array(T[start_idx:end_idx]).reshape((-1, 1))
    data = np.column_stack((data_x, data_y)).astype(np.int32)
    
    timestamp=start_time*np.ones((260,346))
    
    for i in range(0, data.shape[0]):
        timestamp[data[i,1], data[i,0]]=data_T[i]
       
    grayscale = np.flip(255*(timestamp-start_time)/step_time, 0).astype(np.uint8)#The normalization formula

    print(img_count)

    # cv2.imshow('img',grayscale)
   
    # cv2.waitKey(5)

    start_time = end_time
    end_time += step_time
    start_idx = end_idx
    img_count += 1

