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
    def __init__(self, imageAreaFactor = 0.99, numberOfSlidingWindows = 9, windowWidth =100):
        self.imageAreaFactor = imageAreaFactor 
        self.numberOfSlidingWindows = numberOfSlidingWindows
        self.windowWidth = windowWidth 
        self.minimalNumberOfPixels = 50
        
    
    def CalculateHistogram(self,image, yStartValue, yEndValue):
        print(image.shape)
        
        histogram = np.sum(image[yStartValue:yEndValue], axis=0)
        print (len(histogram))
        histogramMidpoint = np.int(histogram.shape[0]/2)
        return histogramMidpoint, histogram
    
    
        
    def Detect(self, processedImage):
        imageHeight = processedImage.shape[0]
        imageWidth = processedImage.shape[1]

        outputImage = np.dstack((processedImage, processedImage, processedImage))*255
        #outputImage = np.dstack((processedImage, processedImage, processedImage))*255
        histogramMidpoint, histogram = self.CalculateHistogram(processedImage, 0, imageHeight)
            
        leftPeakPosition = np.argmax(histogram[:histogramMidpoint])
        rightPeakPosition = histogramMidpoint + np.argmax(histogram[histogramMidpoint:])
            
        currentLeftPeakPosition = leftPeakPosition
        currentRightPeakPosition = rightPeakPosition

        whitePixelPositions = processedImage.nonzero()
        rowIndicesWithWhitePixels = np.array(whitePixelPositions[0])
        columnIndicesWithWhitePixels = np.array(whitePixelPositions[1])

        leftLaneIndices = []
        rightLaneIndices = []
        windowHeight = np.int(imageHeight*self.imageAreaFactor / self.numberOfSlidingWindows)
        
        for windowIndex in range(self.numberOfSlidingWindows):

            windowsBottom = imageHeight - (windowIndex+1)*windowHeight 
            windowsTop = imageHeight - windowIndex*windowHeight 

            leftWindowLeft = currentLeftPeakPosition - self.windowWidth
            leftWindowRight = currentLeftPeakPosition + self.windowWidth
            rightWindowLeft = currentRightPeakPosition - self.windowWidth
            rightWindowRight = currentRightPeakPosition + self.windowWidth
            cv2.rectangle(outputImage,(leftWindowLeft, windowsBottom),(leftWindowRight, windowsTop),   (0,255,0), 2) 
            cv2.rectangle(outputImage,(rightWindowLeft, windowsBottom),(rightWindowRight, windowsTop),   (0,255,0), 2) 
            leftLaneIndexCandidates = ((rowIndicesWithWhitePixels >= windowsBottom) & (rowIndicesWithWhitePixels <= windowsTop) &
                                       (columnIndicesWithWhitePixels >= leftWindowLeft) & (columnIndicesWithWhitePixels <= leftWindowRight)).nonzero()[0]
            rightLaneIndexCandidates = ((rowIndicesWithWhitePixels >= windowsBottom) & (rowIndicesWithWhitePixels <= windowsTop) &
                                       (columnIndicesWithWhitePixels >= rightWindowLeft) & (columnIndicesWithWhitePixels <= rightWindowRight)).nonzero()[0]
            leftLaneIndices.append(leftLaneIndexCandidates)
            rightLaneIndices.append(rightLaneIndexCandidates)
            

            if(len(leftLaneIndexCandidates) > self.minimalNumberOfPixels):
                currentLeftPeakPosition = np.int(np.mean(columnIndicesWithWhitePixels[leftLaneIndexCandidates]))
            

            if(len(rightLaneIndexCandidates) > self.minimalNumberOfPixels):
                currentRightPeakPosition = np.int(np.mean(columnIndicesWithWhitePixels[rightLaneIndexCandidates]))

        
        leftLaneIndices = np.concatenate(leftLaneIndices)
        rightLaneIndices = np.concatenate(rightLaneIndices)
        
        
        leftXValue = columnIndicesWithWhitePixels[leftLaneIndices]
        leftYValue = rowIndicesWithWhitePixels[leftLaneIndices]

        rightXValue = columnIndicesWithWhitePixels[rightLaneIndices]
        rightYValue = rowIndicesWithWhitePixels[rightLaneIndices]
        try:
            leftLaneFit = np.polyfit(leftYValue, leftXValue, 2)
            rightLaneFit = np.polyfit(rightYValue, rightXValue, 2)
            outputImage = self.Visualize(outputImage, leftLaneFit, rightLaneFit, rowIndicesWithWhitePixels, columnIndicesWithWhitePixels, leftLaneIndices, rightLaneIndices)
        except TypeError:
            print("Error ")
            print("Values", leftYValue, leftXValue, rightYValue, rightXValue)
            print("Indices", leftLaneIndices, rightLaneIndices)
        return outputImage
        
    def Visualize(self,outputImage, leftLaneFit, rightLaneFit, rowIndicesWithWhitePixels, columnIndicesWithWhitePixels, leftLaneIndices, rightLaneIndices):
        imageHeight = outputImage.shape[0]
        
        plotY = np.linspace(0 , imageHeight-1, imageHeight)
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
        return outputImage
        
    def DetectAndShow(self, image, warpedImage, processedImage):
        global original
        finalImage = self.Detect(processedImage)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        warpedImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB)
        
        #processedImage = cv2.cvtColor(processedImage, cv2.COLOR_BGR2RGB)
        
        histogramMidpoint, histogram = self.CalculateHistogram(processedImage, 0, processedImage.shape[0])
        #
        f, plots = plt.subplots(2, 2, figsize=(12, 9))
        plots[0,0].imshow(image)
        plots[0,0].set_title('Original ', fontsize=20)
        plots[0,1].imshow(warpedImage)
        plots[0,1].set_title('Warped ', fontsize=20)
        plots[1,0].imshow(processedImage)
        plots[1,0].set_title('Warped and processed ', fontsize=20)
        #plots[1,1].plot(histogram)
        plots[1,1].imshow(finalImage)
        plots[1,1].set_title('Histogram', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

        
        

def TestImagePipeline():
    # Saving, loading data
    data = Data()
    testImages = data.LoadTestImages()
    perspectiveTransform = PerspectiveTransform()
    processing = ImageProcessing(magnitudeKernelSize=11, angleKernelSize=5)
    laneId = LaneLineIdentification()
    if(perspectiveTransform.isCalibrated):
        for image in testImages:
            warpedImage = perspectiveTransform.WarpLaneImage(image)
            processedImage = processing.Process(warpedImage)
            laneId.DetectAndShow(image, warpedImage,processedImage)


def VideoPipeline():
    


TestImagePipeline()
    

 
    




    

    

