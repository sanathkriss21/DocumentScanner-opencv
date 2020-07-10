import cv2 as cv
import  numpy as np


def findwidth_height(initialvalue=480):
    cv.namedWindow("Size")

    cv.createTrackbar("Width", "Size", 480, 2400, nothing)
    cv.createTrackbar("Height", "Size", 480, 2400, nothing)


def getsizeimg():
    Width= cv.getTrackbarPos("Width", "Size")
    Height = cv.getTrackbarPos("Height", "Size")
    src = Width, Height
    return src



def reorder(mypoints):
    mypoints=mypoints.reshape((4,2))
    mypointsnew=np.zeros((4,1,2),dtype=np.int32)
    add=mypoints.sum(1)
    mypointsnew[0] = mypoints[np.argmin(add)]
    mypointsnew[3] = mypoints[np.argmax(add)]
    diff = np.diff(mypoints, axis=1)
    mypointsnew[1] = mypoints[np.argmin(diff)]
    mypointsnew[2] = mypoints[np.argmax(diff)]

    return  mypointsnew

def biggestcont(contours):
    biggest=np.array([])
    max_area=0
    for i in contours:
        area=cv.contourArea(i)


        if (area >8000):
            #print(area)

            peri=cv.arcLength(i,True)
            approx=cv.approxPolyDP(i,0.02*peri,True)
            #print(approx)
            if area > max_area and len(approx)==4:
                 biggest=approx
                 max_area=area
    return biggest,max_area

def drawrectangle(img,biggest,thickness):
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img

def nothing(x):
    pass

def initializetrackbar(intialval=0):
    cv.namedWindow("Trackbar")
    cv.resizeWindow("Trackbar",360,240)
    cv.createTrackbar("Thresh1","Trackbar",200,255,nothing)
    cv.createTrackbar("Thresh2","Trackbar",200,255,nothing)

def valtrackbar():
    thres1=cv.getTrackbarPos("Thresh1","Trackbar")
    thres2=cv.getTrackbarPos("Thresh2","Trackbar")
    src=thres1,thres2
    return src

