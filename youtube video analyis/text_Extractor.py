# -*- coding: utf-8 -*-
"""
Created on Tue May 19 10:51:48 2020

@author: Personal
"""

#C:\Program Files\Tesseract-OCR
import pytesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#from PIL import Image
import glob, os
import imutils
import cv2
import datetime 

#path = './background'
#images = glob.glob('./background\*.jpg')
images = glob.glob('*.jpg')
images.sort(key = os.path.getmtime)
complete_texts = []
i = 0
for image_to_read in images:
    #img = Image.open(image_to_read)
    #img = img.crop((673, 1027, 1515, 1077))
    #img.save("x.jpg")
    im = cv2.imread(image_to_read)
    #im = cv2.imread("x.jpg")
    im = im[1019:1080, 675:1517].copy()
    image = imutils.resize(im, width=1500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.GaussianBlur(thresh, (3,3), 0)
    #img = img.convert('1', dither=Image.NONE)
    
    text = pytesseract.image_to_string(thresh, lang='eng',config='--psm 6')
    complete_texts.append(text)
    i = i+ 1
    print("started {} completed reading image {} of {}".format(image_to_read, i, len(images)))


# for image in images:
#     image = cv2.imread(image)
#     img = image[1019:1067, 675:1517].copy()
    



for t, text in enumerate(complete_texts):
#     #for s, subtexts in enumerate(text.split()):
#     complete_texts[t] = complete_texts[t].replace('f', '7')
#     complete_texts[t] = complete_texts[t].replace('@', '8')
#     complete_texts[t] = complete_texts[t].replace('B', '8')
     complete_texts[t] = complete_texts[t].replace('/', '7')
#     complete_texts[t] = complete_texts[t].replace('Z', '2')
#     complete_texts[t] = complete_texts[t].replace('e', '7')
#     complete_texts[t] = complete_texts[t].replace('@', '8')
#     complete_texts[t] = complete_texts[t].replace('B', '8')
#     complete_texts[t] = complete_texts[t].replace('%', '7')
#     complete_texts[t] = complete_texts[t].replace('SS', '39')
#     complete_texts[t] = complete_texts[t].replace('S', '9')
#     if complete_texts[t][-3] == '9':
#         complete_texts[t] = complete_texts[t].replace('9', 'S')
     subtexts = complete_texts[t].split()
     
     for s, sub in enumerate(subtexts):
         try:
              
             if subtexts[0][3] != 'F':
                subtexts[0] = subtexts[0][:3] #F not needed
                #subtexts[0] = subtexts[0][:3] +"F"
             if subtexts[0][3] == 'F':
                subtexts[0] = subtexts[0][:3] #F not needed
                #subtexts[0] = subtexts[0][:3] +"F"
         except:
                pass
      
         #temparature
         try:
             if s == 0:
                 subtexts[0] = str(subtexts[0]).replace(' ', '')
                 subtexts[0] = str(subtexts[0]).replace('?', '7')
                 subtexts[0] = str(subtexts[0]).replace('/', '7')
                 subtexts[0] = str(subtexts[0]).replace('S', '5')
                 subtexts[0] = str(subtexts[0]).replace('O', '0')
                 subtexts[0] = str(subtexts[0]).replace('f', '7')
                 #subtexts[0] = int(subtexts[0])
         except:
             pass
             #time
         try:
             if s == 1:
                  subtexts.append(subtexts[1])
                  subtexts[1] = str(subtexts[1]).replace(' ', '')
                  subtexts[1].replace("O", "0")
                  if subtexts[2] == "AM":
                      subtexts[1] = subtexts[1][:2] + subtexts[1][3:5]
                      subtexts[1] = int(subtexts[1])
                      print(subtexts[1])
                  if subtexts[2] == "PM":
                      subtexts[1] = int(subtexts[1][:2] + subtexts[1][3:5]) + 1200
                      if subtexts[1] > 2400:
                          subtexts[1] - 1200
                      subtexts[1] = int(subtexts[1])
                  
         except:
             pass
         try:           
             #date 
             if s == 3:
                 subtexts.append(subtexts[3])
                 subtexts[3] = subtexts[3].replace("O", "0")
                 subtexts[3] = datetime.datetime.strptime(subtexts[3], "%m-%d-%y")
                 subtexts[3] = subtexts[3].timetuple().tm_yday
         except:
             pass
             
                 



#             subtexts[0] = str(subtexts).replace('/', '')
#             subtexts[0] = str(subtexts).replace('Z', '2')
#             subtexts[0] = str(subtexts).replace('e', '7')
#             subtexts[0] = str(subtexts).replace('e', '7')
#             subtexts[0] = str(subtexts).replace('@', '8')
#             subtexts[0] = str(subtexts).replace('B', '8')
#             subtexts[0] = str(subtexts).replace('%', '7')
#             subtexts[0] = str(subtexts).replace('SS', '39')
#             subtexts[0] = str(subtexts).replace('S', '9')
                  
     #subtexts = " ".join(subtexts)
     #print(subtexts)
     complete_texts[t] = subtexts
            
              

k =0
with open('witherrors.txt','w')as f:
    for i in complete_texts:
        out = ""
        k+=1
        out = out + str(k) +'\n'
        out += str(i)
        out += '\n\n'
        f.write(out)
        

from xlwt import Workbook 
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1') 
for d, data in enumerate(complete_texts):
    _towrite = data
    sheet1.write(d, 0, d+1)
    sheet1.write(d, 1, _towrite[0])
    sheet1.write(d, 2, _towrite[1])
    sheet1.write(d, 3, _towrite[3]) 
 
    sheet1.write(d, 5, _towrite[4])
    sheet1.write(d, 6, _towrite[5])      
wb.save('towelch.xls')     

