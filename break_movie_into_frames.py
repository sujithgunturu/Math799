import cv2
import numpy as np
import os


FramesPerSecond = 15
captureTheVideo = cv2.VideoCapture('a.mp4')
captureTheVideo.set(cv2.CV_CAP_PROP_FPS, FramesPerSecond)


try:
    if not os.path.exists('frames'):
        os.makedirs('frames')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(True):
    ret, frame = captureTheVideo.read()
    name = './frames/wheat' + str(currentFrame) + '.jpg'
    if not ret: break
    print ('Creating...' + name)
    cv2.imwrite(name, frame)
    currentFrame += 1
        

captureTheVideo.release()
cv2.destroyAllWindows()
