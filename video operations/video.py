# -*- coding: utf-8 -*-
"""
Purpose: To illustrate making a video from a set of jpg files.
         The major elements of the method are drawn from code found at
         https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
         
         and from
         
         https://lorensen.github.io/VTKExamples/site/Python/IO/ImageWriter/
         
Created on Mon Feb 24 09:47:05 2020
@author: welchsm
"""

import cv2
import glob
import os
from time import perf_counter
from natsort import natsorted, ns

def MakeVideo(v_name,F_del=False):
    
    # Obtain the list of images from the file, alphanumeric by file name 
    start = perf_counter()
    img_array = []; fnames=[]
    images = glob.glob('*.jpg')
    images.sort(key = os.path.getmtime)
    #natsorted(images, alg=ns.IGNORECASE)
    for filename in images:
        fnames+=[filename]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        print("rank of image getting added into array", filename)
     
    # Instantiate a video writer    
    out = cv2.VideoWriter(v_name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    print("time taken to add images into array", perf_counter()-start)
    # Make the video
    for i in range(len(img_array)):
        print("writing image", i)
        out.write(img_array[i])
    
    # Release it
    out.release()
    
    # Delete image files if requested
    if F_del:
        for filename in fnames:
            os.remove(filename)
def main():
    MakeVideo("secondmerged")

if __name__ == '__main__':
    main() 
