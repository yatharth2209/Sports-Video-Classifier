import cv2
from skimage.measure import compare_ssim as ssim
import numpy as np
import os
import glob

sport="hurdling"
location1="Predict/"
location2="Data/baseball/predict/"
files=glob.glob(location1+"*.jpg")
for f in files:
	os.remove(f)

files1=glob.glob(location2+"*.jpg")
for f in files1:
	os.remove(f)

#g1=glob.glob()
os.remove("Data/cnn-pred.pkl")

#g2=glob.glob("Data/data.pkl")
os.remove("Data/data.pkl")

def frames(filename):
	#vidcap = cv2.VideoCapture(sport+'/Testing/'+filename)
	vidcap = cv2.VideoCapture('Predict/Input/'+filename)
	success,image = vidcap.read()
	count = 0
	positive=0
	success = True
	while success:
	    success,image = vidcap.read()
	    if success == False:
	    	break
	    print('Read a new frame: ', count)
	    # shutil.rmtree(location1+"*.jpg")
	    # shutil.rmtree(location2+"*.jpg")

	    if count == 0:
	    	#cv2.imwrite(sport+"/Frames/"+filename+"%d.jpg" % count, image)
	    	cv2.imwrite(location1+filename+"%d.jpg" % count, image)
	    	cv2.imwrite(location2+filename+"%d.jpg" % count, image)
	    	prev_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	    	count+=1
	    	positive+=1
	    else:
	        image_copy=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	        s=ssim(image_copy,prev_img)
	        print(s)
	        if s < 0.8: 
	            #cv2.imwrite(sport+"/Frames/"+filename+"%d.jpg" % count, image)
	            cv2.imwrite(location1+filename+"%d.jpg" % count, image)
	            cv2.imwrite(location2+filename+"%d.jpg" % count, image)
	            prev_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	            count += 1
	            positive+=1
	        else:
	            count += 1
	            continue
	print(str(count)+" Frames found, "+str(positive)+" taken.")  

# for f in os.listdir(sport+"/Testing"):
# 	frames(f)

for f in os.listdir("Predict/Input/"):
	frames(f)

