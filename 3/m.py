import cv2
import numpy as np
from collections import Counter
import itertools
from statistics import mode
from pprint import pprint
import sys
import matplotlib.pyplot as plt
import random
import math

radii = []
radiusDelta = 1

# Assuming lines connecting MC options' centers within a question is
centerXDelta = 1
centerYDelta = 1

minCricleW = 16
minCricleH = 16
minCricleArea = ((minCricleW+minCricleH)/4)*((minCricleW+minCricleH)/4)*3
print(minCricleArea)

class mcOption:
    ID = None
    questionID = None
    optionID = None
    centerX = None
    centerY = None
    circleContour = None
    radius = None
    isChecked = False
    centroidID = None
    def __init__(self, ID, questionID, optionID):
        self.ID = ID
        self.questionID = questionID
        self.optionID = optionID
        
    def initCenters(self, circleContour):
        self.circleContour = circleContour
        self.centerX, self.centerY, self.radius = extractFromCricleContour(circleContour)
    
def extractFromCricleContour(circleContour):
    (x, y, w, h) = cv2.boundingRect(circleContour)
    return [x+w/2, y+h/2, w/2]
    
def processImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurry = cv2.GaussianBlur(gray, (3, 3), 1)
    adapt_thresh = cv2.adaptiveThreshold(blurry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return adapt_thresh

def findCircleContours(image):
    processed_image = processImage(image)
    _, contours, hierarchy = cv2.findContours(processed_image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #opencv hierarchy structure: [Next, Previous, First_Child, Parent]
    cv2.drawContours(image, contours,  -1, (0,255,0), 1)
    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    hierarchy = hierarchy[0]
    circleContours = []
    i = 0
    nCirlces = 0
    for contour in contours:       
        (x, y, w, h) = cv2.boundingRect(contour)
        ar = w / float(h)
        if hierarchy[i][3] == -1 and w >= minCricleW and h >= minCricleH and 0.9 <= ar and ar <= 1.2: 
            epsilon = 0.01*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if ( (len(approx) > 8) & (len(approx) < 20) & (area >= minCricleArea) ):
                circleContours.append(contour)
                nCirlces = nCirlces + 1
        i = i + 1
    return [circleContours, nCirlces]

def isCircleChecked(mcOption, image):  # Takes in a mcOption object
    ulx = mcOption.centerX - mcOption.radius
    uly = mcOption.centerY - mcOption.radius
    lrx = mcOption.centerX + mcOption.radius
    lry = mcOption.centerY + mcOption.radius
    nPixels = (lrx-ulx)*(lry-uly)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circleImg = gray_image[int(uly):int(lry),int(ulx):int(lrx)]
    m = cv2.mean(circleImg)
    intensity = m[0]
    #print("--------------------------------->")
    #print(nPixels, m[0])
    if m[0] < 160:
        mcOption.isChecked = True

image = cv2.imread("525.png")
circleContours, nCirlces = findCircleContours(image)

# Initialize a list of objects -------------------->
mcOptions_ObjList = []
for i in range(nCirlces):
    aMcOption = mcOption(i, None, None)
    aMcOption.initCenters(circleContours[i])
    mcOptions_ObjList.append(aMcOption)
    
# Further filter out non-mcOption cirlces --------->    
mcOptions_ObjList = sorted(mcOptions_ObjList , key=lambda k: [k.centerY, k.centerX]) 
for i in range(nCirlces):
    #print(mcOptions_ObjList[i].centerX, mcOptions_ObjList[i].centerY, mcOptions_ObjList[i].radius)
    radii.append(mcOptions_ObjList[i].radius)
radiiMode = mode(radii)
removed = 0
for mcOption in mcOptions_ObjList[:]:
    if not (mcOption.radius - radiusDelta < radiiMode and mcOption.radius + radiusDelta > radiiMode):
        mcOptions_ObjList.remove(mcOption)
        removed = removed + 1
nCirlces = nCirlces - removed
for i in range(nCirlces):
    #print(mcOptions_ObjList[i].centerX, mcOptions_ObjList[i].centerY, mcOptions_ObjList[i].radius)
    isCircleChecked(mcOptions_ObjList[i], image)
    
# Show results ------------------------------------>
del circleContours[:]
for mcOption in mcOptions_ObjList[:]:
    circleContours.append(mcOption.circleContour)
    
cv2.drawContours(image, circleContours,  -1, (0,0,255), 1)
print(nCirlces, "Circles Detected")
cv2.imshow('Circles Detected',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

x  = [mcOption.centerX for mcOption in mcOptions_ObjList]
y  = [mcOption.centerY for mcOption in mcOptions_ObjList]
cx = [mcOption.centerX for mcOption in mcOptions_ObjList if mcOption.isChecked == True]
cy = [mcOption.centerY for mcOption in mcOptions_ObjList if mcOption.isChecked == True]
plt.scatter(x, y, label='MC options dected')
plt.scatter(cx, cy, c='r', label='Checked MC options')
plt.gca().invert_yaxis()
plt.show()

# K-means clustering of questions ----------------->
def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
def average(x, y):
    if sum(x) == 0:
        x = [0, 1]
    if sum(y) == 0:
        y = [0, 1]   
    return [sum(x)/len(x), sum(y)/len(y)]
def createList():
    mylist = []
    for i in range(K): 
        mylist.append(i)
    return mylist
K = 3    # K should be the number of questions.
C = createList()
height, width, _ = image.shape
# Randomly pick K number of centroids
for i in range(0, K):
    C[i] = ([random.randint(0,width), random.randint(0,height)])
isFirstRun = True
while (True):
    input("Press Enter to continue...")
    # Assign each mcOption to the nearest centroid
    nPoints = len(mcOptions_ObjList)
    print(nPoints)
    for i in range(0, nPoints):
        nearestCentroidDistance = 9999999
        for j in range(0, K):
            if (distance([mcOptions_ObjList[i].centerX, mcOptions_ObjList[i].centerY], C[j]) < nearestCentroidDistance):
                nearestCentroidDistance = distance([mcOptions_ObjList[i].centerX, mcOptions_ObjList[i].centerY], C[j])
                mcOptions_ObjList[i].centroidID = j
    print(C)
    for mcOption in mcOptions_ObjList[:]:
        print(mcOption.centroidID)

    # Calculate new centroid for each cluster by taking the mean of the distances between each point and their assigned centroid within that cluster
    for j in range(0, K):
        C[j] = average([mcOption.centerX for mcOption in mcOptions_ObjList if mcOption.centroidID == j], [mcOption.centerY for mcOption in mcOptions_ObjList if mcOption.centroidID == j])
    print(C)
    if (isFirstRun == True):
        lastC = C
    if (isFirstRun == False and lastC == C):
        break
    isFirstRun = False
    lastC = C
