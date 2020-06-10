
import os
from PIL import Image
import glob
from natsort import natsorted, ns

path1 = './frames'
path2 = './background'
newimage = 0
combined = []
new = []

images1 = glob.glob('./frames\*.jpg')
images2 = glob.glob('./background\*.jpg')
natsorted(images1, alg=ns.IGNORECASE)
#natsorted(images2, alg=ns.IGNORECASE)
#images1.sort(key=lambda f: int(filter(str.isdigit, f)))
#images2.sort(key=lambda f: int(filter(str.isdigit, f)))
#images1.sort(key = os.path.getmtime)
images2.sort(key = os.path.getmtime)
i= 0
for filename1, filename2  in zip(images1, images2):
        overlay = Image.open(filename1)
        background = Image.open(filename2)
        newsize = (1920, 1080)                    # change to size of your own
        background = background.resize(newsize)
        overlay = overlay.resize(newsize)
        background = background.convert("RGB")
        overlay = overlay.convert("RGB")
        ov = list(overlay.getdata())
        bg = list(background.getdata())   
        for pix1, pix2 in zip(ov, bg):
                r, g, b = pix1
                if r +g+ b > 20:                     # insert the filtering condition of rgb
                        combined.append(pix1)
                else:
                        combined.append(pix2)
        
        new = Image.new(overlay.mode, overlay.size)
        new.putdata(combined)
        print("super imposing image{} and {}".format(filename1, filename2))
        #new.save("super"+"%03d.jpg"%(i))
        i = i+1
        new.save(str(i) + ".jpg")
        combined = []




        