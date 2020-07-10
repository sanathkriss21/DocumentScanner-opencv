import cv2 as cv
import numpy as np
import utlis

webcamfeed=True
pathImage="pd.jfif"
cap=cv.VideoCapture()
cap.set(10,160)
utlis.findwidth_height()

#height_image=640
#width_image=480

utlis.initializetrackbar()
count=0

while True:
    #if webcamfeed:success,img=cap.read()
    #else:
    img=cv.imread(pathImage)
    #print(img.shape)
    sizeimg=utlis.getsizeimg()
    img=cv.resize(img,(sizeimg[0],sizeimg[1]))
    #print(img.shape)
    imgblank=np.zeros((sizeimg[1],sizeimg[0],3),np.uint8)
    imggray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgblur=cv.GaussianBlur(imggray,(5,5),1)
    #ret, imgthresh = cv.threshold(imgblur, 125, 255, cv.THRESH_BINARY)

    #cv.imshow('thr', imgthresh)

    #cv.imshow('blur',imgblur)
    thres=utlis.valtrackbar()
    imgthresh=cv.Canny(imgblur,thres[0],thres[1])
    #cv.imshow('canny',imgthresh)
    kernel=np.ones((5,5))
    imgdial=cv.dilate(imgthresh,kernel,iterations=2)

    imgthresh=cv.erode(imgdial,kernel,iterations=1)


    imgcon=img.copy()

    imgbcon=img.copy()
    cons,hie=cv.findContours(imgthresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    #print(len(cons))

    #print(hie)
    cv.drawContours(imgcon,cons,-1,(200,250,0),5)



    big,maxarea=utlis.biggestcont(cons)
    #print(maxarea)
    #print(big)
    if big.size !=0:
        big=utlis.reorder(big)
        cv.drawContours(imgbcon,big,-1,(200,250,0),20)
        imgbcon=utlis.drawrectangle(imgbcon,big,2)
        pts1=np.float32(big)
        pts2=np.float32([[0, 0],[sizeimg[0], 0], [0, sizeimg[1]],[sizeimg[0], sizeimg[1]]])
        matrix=cv.getPerspectiveTransform(pts1,pts2)
        imgwrapcolored=cv.warpPerspective(img,matrix,(sizeimg[0], sizeimg[1]))

        #imgwrapcolored=imgwrapcolored[20:imgwrapcolored.shape[0]-20,20:imgwrapcolored.shape[1]-20]
        imgwrapcolored=cv.resize(imgwrapcolored,(sizeimg[0], sizeimg[1]))

        imgwrapgray=cv.cvtColor(imgwrapcolored,cv.COLOR_BGR2GRAY)
        imgadaptive=cv.adaptiveThreshold(imgwrapgray,255,1,1,7,2)
        imgadaptive=cv.bitwise_not(imgadaptive)
        imgadaptive=cv.medianBlur(imgadaptive,3)
        cv.imshow('document', imgadaptive)
        imgarray=([img,imggray,imgadaptive,imgcon],
                  [imgbcon,imgwrapcolored,imgwrapgray,imgadaptive])
    else:
        imgarray = ([img, imggray, imgthresh, imgcon],
                   [imgblank,imgblank,imgblank,imgblank])


    #print(len(imgarray))
    #stackedimg=utlis.stackimg(imgarray,0.75,lables)

    cv.imshow("Final",imgadaptive)





    k= cv.waitKey(1)
    if k==ord('s'):
        cv.imwrite("Scanned" + str(count) + ".jpg", imgwrapcolored)
        cv.putText(imgadaptive,"Scanned",(int(imgadaptive.shape[1]/2 )-200,
                                         int(imgadaptive.shape[0]/2)),cv.FONT_HERSHEY_SIMPLEX,2,(0,250,0),5,cv.LINE_AA)

        cv.imshow('Result', imgadaptive)
        cv.waitKey(300)
        count += 1




