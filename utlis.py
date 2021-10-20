import cv2
import numpy as np

def getContours(img, cThr= [100, 100], showCanny=False, minArea=1000, filter=0, draw = False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernal = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernal, iterations=3)
    imgThre = cv2.erode(imgDial,  kernal, iterations=2)
    if showCanny:cv2.imshow('Canny', imgThre)

    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key = lambda x:x[1], reverse=True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    return img, finalCountours

def reorder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    #print(myPoints)
    add = myPoints.sum(1)
    #print(add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    #print(myPointsNew[0])
    myPointsNew[3] = myPoints[np.argmax(add)]
    #print(myPointsNew[3])
    diff = np.diff(myPoints, axis=1)
    #print(diff)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    #print(myPointsNew[1])
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print(myPointsNew[2])
    return myPointsNew

def warpImg(img, points, w, h, pad=20):
    #print(points)
    points = reorder(points)
    #print(points)
    pts1 = np.float32(points)
    #print(pts1)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    #print(pts2)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    #print(matrix)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    #print(imgWarp)

    return imgWarp

def finDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

