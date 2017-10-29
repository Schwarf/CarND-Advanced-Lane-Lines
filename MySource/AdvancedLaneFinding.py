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
        self.areLanesDetected = False
        self.windowWidth = windowWidth 
        self.minimalNumberOfPixels = 50
        self.rowIndicesWithWhitePixels = None
        self.columnIndicesWithWhitePixels = None
        self.leftLaneFit = None
        self.rightLaneFit = None
    
    def CalculateHistogram(self,image, yStartValue, yEndValue):
        histogram = np.sum(image[yStartValue:yEndValue], axis=0)
        histogramMidpoint = np.int(histogram.shape[0]/2)
        return histogramMidpoint, histogram
    
    
    def FindWhitePixels(self, image):
        whitePixelPositions = image.nonzero()
        self.rowIndicesWithWhitePixels = np.array(whitePixelPositions[0])
        self.columnIndicesWithWhitePixels = np.array(whitePixelPositions[1])

    
    def UpdateQuadraticFit(self, xValue, yValue):
        return np.polyfit(yValue, xValue, 2)
    
    
    def GetLeftLaneFitForYValues(self, yValues):
        return self.leftLaneFit[0]*yValues**2 + self.leftLaneFit[1]*yValues + self.leftLaneFit[2]

    def GetRightLaneFitForYValues(self, yValues):
        return self.rightLaneFit[0]*yValues**2 + self.rightLaneFit[1]*yValues + self.rightLaneFit[2]


        
    def DetectLanes(self, processedImage):
        if self.areLanesDetected:
            self.IterateFromDetectedLanes(processedImage)
            return
        print ("Lane detection is runnning ... ")
        imageHeight = processedImage.shape[0]
        imageWidth = processedImage.shape[1]

        outputImage = np.dstack((processedImage, processedImage, processedImage))*255

        histogramMidpoint, histogram = self.CalculateHistogram(processedImage, 0, imageHeight)
            
        leftPeakPosition = np.argmax(histogram[:histogramMidpoint])
        rightPeakPosition = histogramMidpoint + np.argmax(histogram[histogramMidpoint:])
            
        currentLeftPeakPosition = leftPeakPosition
        currentRightPeakPosition = rightPeakPosition

        self.FindWhitePixels(processedImage)

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
            
            leftLaneIndexCandidates = ((self.rowIndicesWithWhitePixels >= windowsBottom) & (self.rowIndicesWithWhitePixels <= windowsTop) &
                                       (self.columnIndicesWithWhitePixels >= leftWindowLeft) & (self.columnIndicesWithWhitePixels <= leftWindowRight)).nonzero()[0]
            
            rightLaneIndexCandidates = ((self.rowIndicesWithWhitePixels >= windowsBottom) & (self.rowIndicesWithWhitePixels <= windowsTop) &
                                       (self.columnIndicesWithWhitePixels >= rightWindowLeft) & (self.columnIndicesWithWhitePixels <= rightWindowRight)).nonzero()[0]
            
            
            leftLaneIndices.append(leftLaneIndexCandidates)
            rightLaneIndices.append(rightLaneIndexCandidates)
            

            if(len(leftLaneIndexCandidates) > self.minimalNumberOfPixels):
                currentLeftPeakPosition = np.int(np.mean(self.columnIndicesWithWhitePixels[leftLaneIndexCandidates]))
            

            if(len(rightLaneIndexCandidates) > self.minimalNumberOfPixels):
                currentRightPeakPosition = np.int(np.mean(self.columnIndicesWithWhitePixels[rightLaneIndexCandidates]))

        
        leftLaneIndices = np.concatenate(leftLaneIndices)
        rightLaneIndices = np.concatenate(rightLaneIndices)
        
        
        leftXValue = self.columnIndicesWithWhitePixels[leftLaneIndices]
        leftYValue = self.rowIndicesWithWhitePixels[leftLaneIndices]

        rightXValue = self.columnIndicesWithWhitePixels[rightLaneIndices]
        rightYValue = self.rowIndicesWithWhitePixels[rightLaneIndices]
        try:
            self.leftLaneFit =  self.UpdateQuadraticFit(leftXValue, leftYValue)  #np.polyfit(leftYValue, leftXValue, 2)
            self.rightLaneFit = self.UpdateQuadraticFit(rightXValue, rightYValue)  #np.polyfit(rightYValue, rightXValue, 2)
            
            outputImage = self.Visualize(outputImage, leftLaneIndices, rightLaneIndices)
            self.areLanesDetected = True
            print ("... lane detection complete.")
        except TypeError:
            print("Error ")
            print("Values", leftYValue, leftXValue, rightYValue, rightXValue)
            print("Indices", leftLaneIndices, rightLaneIndices)
        return outputImage
        
    def Visualize(self,outputImage, leftLaneIndices, rightLaneIndices):
        imageHeight = outputImage.shape[0]
        
        plotY = np.linspace(0 , imageHeight-1, imageHeight)
        leftLaneFitX = self.GetLeftLaneFitForYValues(plotY)
        rightLaneFitX = self.GetRightLaneFitForYValues(plotY)
        outputImage[self.rowIndicesWithWhitePixels[leftLaneIndices], self.columnIndicesWithWhitePixels[leftLaneIndices]] = [255,0,0]
        outputImage[self.rowIndicesWithWhitePixels[rightLaneIndices], self.columnIndicesWithWhitePixels[rightLaneIndices]] = [0,0,255]
        
        plt.imshow(outputImage)
        plt.plot(leftLaneFitX, plotY, color='yellow')
        plt.plot(rightLaneFitX, plotY, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        return outputImage
    
    
    def IterateFromDetectedLanes(self, processedImage):
        imageHeight = processedImage.shape[0]
        imageWidth = processedImage.shape[1]
        outputImage = np.dstack((processedImage, processedImage, processedImage))*255
        
        self.FindWhitePixels(processedImage)
        
        leftLaneCondition = self.GetLeftLaneFitForYValues(self.rowIndicesWithWhitePixels)  
        rightLaneCondition = self.GetRightLaneFitForYValues(self.rowIndicesWithWhitePixels)

        leftLaneIndices = ((self.columnIndicesWithWhitePixels > (leftLaneCondition- self.windowWidth)) & (self.columnIndicesWithWhitePixels < (leftLaneCondition + self.windowWidth)))
        rightLaneIndices = ((self.columnIndicesWithWhitePixels > (rightLaneCondition- self.windowWidth)) & (self.columnIndicesWithWhitePixels < (rightLaneCondition + self.windowWidth)))

        leftXValue = self.columnIndicesWithWhitePixels[leftLaneIndices]
        leftYValue = self.rowIndicesWithWhitePixels[leftLaneIndices]

        rightXValue = self.columnIndicesWithWhitePixels[rightLaneIndices]
        rightYValue = self.rowIndicesWithWhitePixels[rightLaneIndices]

        self.leftLaneFit =  self.UpdateQuadraticFit(leftXValue, leftYValue)  #np.polyfit(leftYValue, leftXValue, 2)
        self.rightLaneFit = self.UpdateQuadraticFit(rightXValue, rightYValue)  #np.polyfit(rightYValue, rightXValue, 2)
        
        
        outputImage = self.Visualize(outputImage, leftLaneIndices, rightLaneIndices)
        

        
    def DetectAndShow(self, image, warpedImage, processedImage):
        global original
        finalImage = self.DetectLanes(processedImage)
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



def VideoImageProcessing(image):
    """
    Init
    """
    processing = ImageProcessing(magnitudeKernelSize=11, angleKernelSize=5)
    perspectiveTransform = PerspectiveTransform()
    laneId = LaneLineIdentification()
    
    warpedImage = perspectiveTransform.WarpLaneImage(image)
    processedImage = processing.Process(warpedImage)
    laneId.DetectAndShow(image, warpedImage,processedImage)

        

def TestImagePipeline():
    # Saving, loading data
    data = Data()
    testImages = data.LoadTestImages()
    perspectiveTransform = PerspectiveTransform()
    processing = ImageProcessing(magnitudeKernelSize=11, angleKernelSize=5)
    laneId = LaneLineIdentification()
    if(perspectiveTransform.isCalibrated):
        for image in testImages:
            laneId.areLanesDetected = False
            warpedImage = perspectiveTransform.WarpLaneImage(image)
            processedImage = processing.Process(warpedImage)
            laneId.DetectAndShow(image, warpedImage,processedImage)


def VideoPipeline():
    data =Data()
    projectVideo = data.LoadProjectVideo()
    output = projectVideo.fl_image(lambda x:  VideoImageProcessing(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))) 
    


#TestImagePipeline()
VideoPipeline()   

 
    




    

    

