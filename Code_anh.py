import cv2
import pytesseract
from PIL import Image, ImageEnhance
import numpy as np
import Preprocess
import math
import tensorflow as tf
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image

ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

n = 1

Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 25
RESIZED_IMAGE_HEIGHT = 55


img = cv2.imread("Biensoxe/9.jpg")
img = cv2.resize(img,dsize = (1366,768))

################ 
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
canny_image = cv2.Canny(imgThreshplate,250,255) 
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1) 

###### 
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10] 

screenCnt = []
for c in contours:
    peri = cv2.arcLength(c, True) 
    approx = cv2.approxPolyDP(c, 0.07 * peri, True) 
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w/h

    if (len(approx) == 4) :
        screenCnt.append(approx)  
        
        #cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3) 

if screenCnt is None:
    detected = 0
    print ("No plate detected")
else:
    detected = 1

if detected == 1:

    for screenCnt in screenCnt:
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3) 
        
        ############## 
        (x1,y1) = screenCnt[0,0]
        (x2,y2) = screenCnt[1,0]
        (x3,y3) = screenCnt[2,0]
        (x4,y4) = screenCnt[3,0]
        array = [[x1, y1], [x2,y2], [x3,y3], [x4,y4]]
        sorted_array = array.sort(reverse=True, key=lambda x:x[1])
        (x1,y1) = array[0]
        (x2,y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs (x1 - x2)
        angle = math.atan(doi/ke) * (180.0 / math.pi)

        ########## 
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
        ###
        (x, y) = np.where(mask == 255)       
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx)/2, (bottomy - topy)/2

        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx ))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx ))
        roi = cv2.resize(roi,(0,0),fx = 3, fy = 3)
        imgThresh = cv2.resize(imgThresh,(0,0),fx = 3, fy = 3)

        #################### 
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        thre_mor = cv2.morphologyEx(imgThresh,cv2.MORPH_DILATE,kerel3)
        cont,hier = cv2.findContours(thre_mor,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 

        cv2.imshow(str(n+20),thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2) 

        ##################### 
        char_x_ind = {}
        char_x = []
        height, width,_ = roi.shape
        roiarea = height * width

        for ind,cnt in enumerate(cont) :
            (x,y,w,h) = cv2.boundingRect(cont[ind])
            ratiochar = w/h
            char_area = w*h

            if (Min_char*roiarea < char_area < Max_char*roiarea) and ( 0.25 < ratiochar < 0.7):
                if x in char_x:
                    x = x + 1
                char_x.append(x)    
                char_x_ind[x] = ind
                 
        ############ 

        char_x = sorted(char_x)
        strFinalString = ""
        first_line = ""
        second_line = ""

        for i in char_x:
            (x,y,w,h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
        
            imgROI = thre_mor[y:y+h,x:x+w]     # cắt kí tự ra khỏi hình
            
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize lại hình ảnh
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # đưa hình ảnh về mảng 1 chiều
            cv2.imwrite("Kytu.jpg",imgROIResized)

            model=load_model('modellll.h5')
            path="Kytu.jpg"     
            img_file=tf.keras.utils.load_img(path,target_size=(25,55))
            x=tf.keras.utils.img_to_array(img_file)
            x=np.expand_dims(x, axis=0)
            img_data=preprocess_input(x)
            classes=model.predict(img_data)

            #print("So " + str(classes) + "\n")
            if int(classes[0][0])==1:
                strCurrentChar='0'
            elif int(classes[0][1])==1:
                strCurrentChar='1'
            elif int(classes[0][2])==1:
                strCurrentChar='2'
            elif int(classes[0][3])==1:
                strCurrentChar='3'
            elif int(classes[0][4])==1:
                strCurrentChar='4'
            elif int(classes[0][5])==1:
                strCurrentChar='5'
            elif int(classes[0][6])==1:
                strCurrentChar='6'
            elif int(classes[0][7])==1:
                strCurrentChar='7'
            elif int(classes[0][8])==1:
                strCurrentChar='8'
            elif int(classes[0][9])==1:
                strCurrentChar='9'
            elif int(classes[0][10])==1:
                strCurrentChar='A'
            elif int(classes[0][11])==1:
                strCurrentChar='B'
            elif int(classes[0][12])==1:
                strCurrentChar='C'
            elif int(classes[0][13])==1:
                strCurrentChar='D'
            elif int(classes[0][14])==1:
                strCurrentChar='E'
            elif int(classes[0][15])==1:
                strCurrentChar='F'
            elif int(classes[0][16])==1:
                strCurrentChar='G'
            elif int(classes[0][17])==1:
                strCurrentChar='H'
            elif int(classes[0][18])==1:
                strCurrentChar='K'
            elif int(classes[0][19])==1:
                strCurrentChar='L'
            elif int(classes[0][20])==1:
                strCurrentChar='M'
            elif int(classes[0][21])==1:
                strCurrentChar='N'
            elif int(classes[0][22])==1:
                strCurrentChar='P'
            elif int(classes[0][23])==1:
                strCurrentChar='Q'
            elif int(classes[0][24])==1:
                strCurrentChar='R'
            elif int(classes[0][25])==1:
                strCurrentChar='S'
            elif int(classes[0][26])==1:
                strCurrentChar='T'
            elif int(classes[0][27])==1:
                strCurrentChar='U'
            elif int(classes[0][28])==1:
                strCurrentChar='V'
            elif int(classes[0][29])==1:
                strCurrentChar='X'
            elif int(classes[0][30])==1:
                strCurrentChar='Y'
            elif int(classes[0][31])==1:
                strCurrentChar='Z'

            if (y < height/3): # Biển số 1 hay 2 hàng
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar
                
        print ("\n License Plate " +str(n)+ " is: " + first_line + " - " + second_line + "\n")
        
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75) 
        cv2.imshow(str(n),cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        
        if first_line == "63B1":
            print("Huyen Cai Be - Tien Giang")
        if first_line == "63B2":
            print("Thi Xa Cai Lay - Tien Giang")
        if first_line == "63B3":
            print("Huyen Chau Thanh - Tien Giang")
        if first_line == "63B4":
            print("Huyen Cho Gao - Tien Giang")
        if first_line == "63B5":
            print("Huyen Go Cong Tay - Tien Giang")
        if first_line == "63B6":
            print("Thi Xa Go Cong - Tien Giang")
        if first_line == "63B7":
            print("Huyen Go Cong Dong - Tien Giang")
        if first_line == "63B8":
            print("Huyen Tan Phu Dong - Tien Giang")
        if first_line == "63B9":
            print("Thanh pho My Tho - Tien Giang")
        if first_line == "63P1":
            print("Huyen Cai Lay - Tien Giang")

        cv2.putText(img, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        n = n + 1
        


img = cv2.resize(img, None, fx=0.5, fy=0.5) 
cv2.imshow('License plate', img)

cv2.waitKey(0)


