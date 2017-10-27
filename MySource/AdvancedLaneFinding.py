'''
Created on Oct 25, 2017

@author: andre
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


from ImageProcessing import ImageProcessing
from DataHandling import Data
from PerspectiveTransform import  PerspectiveTransform
 







class LaneLineIdentification():
    def __init__(self, imageAreaFactor = 2, numberOfSlidingWindows = 9, windowWidth =100):
        self.imageAreaFactor = imageAreaFactor 
        self.numberOfSlidingWindows = numberOfSlidingWindows
        self.windowWidth = windowWidth 
        self.minimalNumberOfPixels = 50
        self.imageBottomOffset = 100
    
    def CalculateHistogram(self, image):
        histogram = np.sum(image[image.shape[0]//self.imageAreaFactor -self.imageBottomOffset:,:], axis=0)
        histogramMidpoint = np.int(histogram.shape[0]/2)
        return histogramMidpoint, histogram
    
    
        
    def Detect(self, warpedImage):
        imageHeight = warpedImage.shape[0]
        imageWidth = warpedImage.shape[1]

        outputImage = np.dstack((warpedImage, warpedImage, warpedImage))*255
        #outputImage = np.dstack((processedImage, processedImage, processedImage))*255
        histogramMidpoint, histogram = self.CalculateHistogram(warpedImage)
        #histogramMidpoint, histogram = self.CalculateHistogram(processedImage)
        leftPeakPosition = np.argmax(histogram[:histogramMidpoint])
        rightPeakPosition = histogramMidpoint + np.argmax(histogram[histogramMidpoint:])
        
        whitePixelPositions = warpedImage.nonzero()
        rowIndicesWithWhitePixels = np.array(whitePixelPositions[0])
        columnIndicesWithWhitePixels = np.array(whitePixelPositions[1])

        currentLeftPeakPosition = leftPeakPosition
        currentRightPeakPosition = rightPeakPosition
        leftLaneIndices = []
        rightLaneIndices = []
        windowHeight = np.int(imageHeight//self.imageAreaFactor / self.numberOfSlidingWindows)
        
        for windowIndex in range(self.numberOfSlidingWindows):
            windowsBottom = imageHeight - (windowIndex+1)*windowHeight - self.imageBottomOffset
            windowsTop = imageHeight - windowIndex*windowHeight - self.imageBottomOffset
            
            leftWindowLeft = leftPeakPosition - self.windowWidth
            leftWindowRight = leftPeakPosition + self.windowWidth
            rightWindowLeft = rightPeakPosition - self.windowWidth
            rightWindowRight = rightPeakPosition + self.windowWidth
            
            leftLaneIndexCandidates = ((rowIndicesWithWhitePixels >= windowsBottom) & (rowIndicesWithWhitePixels <= windowsTop) &
                                       (columnIndicesWithWhitePixels >= leftWindowLeft) & (columnIndicesWithWhitePixels <= leftWindowRight)).nonzero()[0]
            rightLaneIndexCandidates = ((rowIndicesWithWhitePixels >= windowsBottom) & (rowIndicesWithWhitePixels <= windowsTop) &
                                       (columnIndicesWithWhitePixels >= rightWindowLeft) & (columnIndicesWithWhitePixels <= rightWindowRight)).nonzero()[0]
            #print (windowsTop, windowsBottom, leftWindowLeft, leftWindowRight)
            #print(rowIndicesWithWhitePixels[200:310], rowIndicesWithWhitePixels[200:310] >windowsBottom )
            leftLaneIndices.append(leftLaneIndexCandidates)
            rightLaneIndices.append(rightLaneIndexCandidates)
            
            if(len(leftLaneIndices) > self.minimalNumberOfPixels):
                currentLeftPeakPosition = np.int(np.mean(columnIndicesWithWhitePixels[leftLaneIndexCandidates]))
            if(len(rightLaneIndices) > self.minimalNumberOfPixels):
                currentRightPeakPosition = np.int(np.mean(columnIndicesWithWhitePixels[rightLaneIndexCandidates]))
        
        leftLaneIndices = np.concatenate(leftLaneIndices)
        rightLaneIndices = np.concatenate(rightLaneIndices)
        
        
        leftXValue = columnIndicesWithWhitePixels[leftLaneIndices]
        leftYValue = rowIndicesWithWhitePixels[leftLaneIndices]

        rightXValue = columnIndicesWithWhitePixels[rightLaneIndices]
        rightYValue = rowIndicesWithWhitePixels[rightLaneIndices]
        
        leftLaneFit = np.polyfit(leftYValue, leftXValue, 2)
        rightLaneFit = np.polyfit(rightYValue, rightXValue, 2)
        self.Visualize(outputImage, leftLaneFit, rightLaneFit, rowIndicesWithWhitePixels, columnIndicesWithWhitePixels, leftLaneIndices, rightLaneIndices)
    
    def Visualize(self,outputImage, leftLaneFit, rightLaneFit, rowIndicesWithWhitePixels, columnIndicesWithWhitePixels, leftLaneIndices, rightLaneIndices):
        imageHeight = outputImage.shape[0]
        
        plotY = np.linspace(imageHeight-imageHeight//self.imageAreaFactor - self.imageBottomOffset, imageHeight - self.imageBottomOffset, imageHeight//2)
        leftLaneFitX = leftLaneFit[0]*plotY**2 + leftLaneFit[1]*plotY + leftLaneFit[2]
        rightLaneFitX = rightLaneFit[0]*plotY**2 + rightLaneFit[1]*plotY + rightLaneFit[2]
        outputImage[rowIndicesWithWhitePixels[leftLaneIndices], columnIndicesWithWhitePixels[leftLaneIndices]] = [255,0,0]
        outputImage[rowIndicesWithWhitePixels[rightLaneIndices], columnIndicesWithWhitePixels[rightLaneIndices]] = [0,0,255]
        plt.imshow(outputImage)
        plt.plot(leftLaneFitX, plotY, color='yellow')
        plt.plot(rightLaneFitX, plotY, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        
        
    def DetectAndShow(self, image, warpedImage, processedImage):
        global original
        #if(original):
        #    self.Detect(processedImage)
            #warpedImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        warpedImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB)
        
        #processedImage = cv2.cvtColor(processedImage, cv2.COLOR_BGR2RGB)
        
        histogramMidpoint, histogram = self.CalculateHistogram(processedImage)
        #
        f, plots = plt.subplots(2, 2, figsize=(12, 9))
        plots[0,0].imshow(image)
        plots[0,0].set_title('Original ', fontsize=20)
        plots[0,1].imshow(warpedImage)
        plots[0,1].set_title('Warped ', fontsize=20)

        plots[1,0].imshow(processedImage)
        plots[1,0].set_title('Warped and processed ', fontsize=20)
        plots[1,1].plot(histogram)
        plots[1,1].set_title('Histogram', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

        
        






    
data = Data()
'''
Load the data
'''
chessBoardImages = data.LoadChessBoardImages()
testImages = data.LoadTestImages()

'''
Initialize perspective transformation class 
'''
perspectiveTransform = PerspectiveTransform()

'''
Initialize image processing 
'''

processing = ImageProcessing(magnitudeKernelSize=11, angleKernelSize=5)


'''
Lane detection
'''

laneId = LaneLineIdentification()
original = True
if(perspectiveTransform.isCalibrated):
    for image in testImages:
        
        if(original):
            warpedImage = perspectiveTransform.WarpLaneImage(image)
            processedImage = processing.Process(warpedImage)
        else:
            processedImage = processing.Process(image)
            warpedImage = perspectiveTransform.WarpLaneImage(processedImage)
            
        
        laneId.DetectAndShow(image, warpedImage,processedImage)
        

#testImages = data.LoadTestImages()

#for image in testImages:
#    Processing.ShowProcessedImage(image)

 

    

