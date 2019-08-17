import sys
import os
import math
import cv2
import numpy as np
import argparse
import dvsproc

# parse the command line argument
parser = argparse.ArgumentParser(description='Frequency for encoding.')
parser.add_argument('file_path', 
                help='The .aedat file path.')
args = parser.parse_args()

inputfile = args.file_path

T, X, Y, Pol = dvsproc.getDVSeventsDavis(inputfile)#Read the quaternion array
T = np.array(T).reshape((-1, 1))
X = np.array(X).reshape((-1, 1))
Y = np.array(Y).reshape((-1, 1))
Pol = np.array(Pol).reshape((-1, 1))

step_time = 10000 #The cumulative time of a frame

start_idx = 0
end_idx = 0
start_time = T[0]
print(start_time)
end_time = start_time + step_time
img_count = 0

while end_time <= T[-1]:
   
    while T[end_idx] < end_time:
        end_idx = end_idx + 1
    
    data_x = np.array(X[start_idx:end_idx]).reshape((-1, 1))
    data_y = np.array(Y[start_idx:end_idx]).reshape((-1, 1))
    data = np.column_stack((data_x, data_y)).astype(np.int32)
   
    counter=np.zeros((260,346))
    
    for i in range(0, data.shape[0]):
        counter[data[i,1], data[i,0]]+=1#Count the number of pixel occurrences
      
    grayscale = np.flip(255*2*(1/(1+np.exp(-counter))-0.5),0)#The normalization formula
   
    cv2.imshow('img',grayscale)
    
    cv2.waitKey(5)
    # wfile='D:/ms/' + str(img_count) + '.png'
    # cv2.imwrite(wfile,grayscale)

    start_time = end_time
    end_time += step_time
    start_idx = end_idx
    img_count += 1
    