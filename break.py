import cv2
vidcap = cv2.VideoCapture('fullpart.mp4')
import os
try:
    if not os.path.exists('backgroud second part'):
        os.makedirs('backgroud second part')
except OSError:
    print ('Error: Creating directory of data')
i = 0 
def getFrame(sec):
    global i
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        i = i +1 
        print('creating frame number ' + str(i) )
        cv2.imwrite("./backgroud second part/background "+str(i)+".jpg", image)
    return hasFrames
sec = 0
fps = 15
frameRate = 1/15
success = getFrame(sec)
while success:
    sec = sec + frameRate
   # sec = round(sec, 2)
    success = getFrame(sec)