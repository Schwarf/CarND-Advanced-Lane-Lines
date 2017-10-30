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
from ImageTransform import  ImageTransform
from collections import deque

BigCounter = 0


ShowIt = False



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
        self.averagedLeftLaneList = deque()
        self.averagedRightLaneList = deque()
        
    
    def CalculateHistogram(self,image, yStartValue, yEndValue):
        histogram = np.sum(image[yStartValue:yEndValue], axis=0)
        histogramMidpoint = np.int(histogram.shape[0]/2)
        return histogramMidpoint, histogram
    
    
    def FindWhitePixels(self, image):
        whitePixelPositions = image.nonzero()
        self.rowIndicesWithWhitePixels = np.array(whitePixelPositions[0])
        self.columnIndicesWithWhitePixels = np.array(whitePixelPositions[1])

    
    def UpdateQuadraticFit(self, xValue, yValue):
        global BigCounter
        BigCounter +=1
        return np.polyfit(yValue, xValue, 2)
    
    
    def GetLaneForYValues(self, fit, yValues, valid, average, nextFrameFit =False):
        if nextFrameFit:
            return  fit[0]*yValues**2 + fit[1]*yValues + fit[2], None
        
        count = len(average)
        if(valid):
            result = fit[0]*yValues**2 + fit[1]*yValues + fit[2]
            if(count > 0):
                averageValue = sum(average)/float(count)
                print (np.divide( (result -averageValue)**2,averageValue**2))
                normalizedMeanSquareDeviation = np.mean(np.divide( (result -averageValue)**2,averageValue**2))
                if(normalizedMeanSquareDeviation < 0.072):
                    average.append(result)
                else:
                    result = averageValue
            else:
                average.append(result)
        
        
        else:        
            result = sum(average)/float(count)
        if(count > 15):
            average.popleft()

        return result, average
    
    

        
        
        
    def DetectLanes(self, processedImage):
        if self.areLanesDetected:
            laneImage = self.IterateFromDetectedLanes(processedImage)
            return laneImage
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
        
        isLeftValid, leftXValues, leftYValues =  self.CalculateXYValues(leftLaneIndices)
        isRightValid, rightXValues, rightYValues =  self.CalculateXYValues( rightLaneIndices)

        
        
        if isLeftValid :
            self.leftLaneFit =  self.UpdateQuadraticFit(leftXValues, leftYValues)  
        if isRightValid:
            self.rightLaneFit = self.UpdateQuadraticFit(rightXValues, rightYValues)  
        
        if isLeftValid and isRightValid:
            self.areLanesDetected = True
        
#        if not self.areLanesDetected:
#            print(len(self.averagedRightLaneList) , len(self.averagedLeftLaneList), isRightValid )
#            outputImage = self.Visualize(outputImage, leftLaneIndices, rightLaneIndices, isLeftValid, isRightValid)
        outputImage = self.VisualizeLane(processedImage, isLeftValid, isRightValid)
        return outputImage
        
    def Visualize(self,outputImage, leftLaneIndices, rightLaneIndices, isLeftValid, isRightValid):
        imageHeight = outputImage.shape[0]
        
        yValues = np.linspace(0 , imageHeight-1, imageHeight)
        leftLaneFitX, self.averagedLeftLaneList = self.GetLaneForYValues(self.leftLaneFit, yValues, isLeftValid, self.averagedLeftLaneList)
        rightLaneFitX, self.averageRightLaneList = self.GetLaneForYValues(self.rightLaneFit, yValues, isRightValid, self.averagedRightLaneList)
        leftPoints = np.array([np.transpose(np.vstack([leftLaneFitX, yValues]))])
        rightPoints = np.array([np.flipud(np.transpose(np.vstack([rightLaneFitX, yValues])))])
        allPoints = np.hstack((leftPoints, rightPoints))
        cv2.fillPoly(outputImage, np.int_([allPoints]), (0,255, 0))
        outputImage[self.rowIndicesWithWhitePixels[leftLaneIndices], self.columnIndicesWithWhitePixels[leftLaneIndices]] = [255,0,0]
        outputImage[self.rowIndicesWithWhitePixels[rightLaneIndices], self.columnIndicesWithWhitePixels[rightLaneIndices]] = [0,0,255]
        
        plt.imshow(outputImage)
        plt.plot(leftLaneFitX, yValues, color='yellow')
        plt.plot(rightLaneFitX, yValues, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
        return outputImage
    
    
    def VisualizeLane(self, processedImage, isLeftValid, isRightValid):
        imageHeight = processedImage.shape[0]
        imageWidth = processedImage.shape[1]
        outputImage = np.dstack((processedImage, processedImage, processedImage))*255

        yValues = np.linspace(0 , imageHeight-1, imageHeight)
        leftLaneFitX, self.averagedLeftLaneList = self.GetLaneForYValues(self.leftLaneFit, yValues, isLeftValid, self.averagedLeftLaneList)
        rightLaneFitX, self.averagedRightLaneList = self.GetLaneForYValues(self.rightLaneFit, yValues, isRightValid, self.averagedRightLaneList)
            
        leftPoints = np.array([np.transpose(np.vstack([leftLaneFitX, yValues]))])
        rightPoints = np.array([np.flipud(np.transpose(np.vstack([rightLaneFitX, yValues])))])
        allPoints = np.hstack((leftPoints, rightPoints))
        
        cv2.fillPoly(outputImage, np.int_([allPoints]), (0,255, 0))
        return outputImage
    
    def CalculateXYValues(self, indices):
        isValid = True
        xValues = self.columnIndicesWithWhitePixels[indices]
        yValues = self.rowIndicesWithWhitePixels[indices]
        pixelCount = 100
        if(len(xValues) < pixelCount or len(yValues)< pixelCount):
            isValid = False
        return isValid, xValues, yValues
         
    
    def IterateFromDetectedLanes(self, processedImage):
        self.FindWhitePixels(processedImage)
        #outputImage = np.dstack((processedImage, processedImage, processedImage))*255
        leftLaneCondition, _ = self.GetLaneForYValues(self.leftLaneFit, self.rowIndicesWithWhitePixels, True, deque(), nextFrameFit=True) 
        rightLaneCondition, _ = self.GetLaneForYValues(self.rightLaneFit, self.rowIndicesWithWhitePixels, True, deque(), nextFrameFit=True)
        
        leftLaneIndices = ((self.columnIndicesWithWhitePixels > (leftLaneCondition- self.windowWidth)) & (self.columnIndicesWithWhitePixels < (leftLaneCondition + self.windowWidth)))
        rightLaneIndices = ((self.columnIndicesWithWhitePixels > (rightLaneCondition- self.windowWidth)) & (self.columnIndicesWithWhitePixels < (rightLaneCondition + self.windowWidth)))

        isLeftValid, leftXValues, leftYValues =  self.CalculateXYValues(leftLaneIndices)
        isRightValid, rightXValues, rightYValues =  self.CalculateXYValues( rightLaneIndices)
        
        
        self.areLanesDetected =False
        if isLeftValid :
            self.leftLaneFit =  self.UpdateQuadraticFit(leftXValues, leftYValues)  
        if isRightValid:
            self.rightLaneFit = self.UpdateQuadraticFit(rightXValues, rightYValues)  
        
        if isLeftValid and isRightValid:
            self.areLanesDetected = True

        #outputImage = self.Visualize(outputImage, leftLaneIndices, rightLaneIndices, isLeftValid, isRightValid)        
        outputImage = self.VisualizeLane(processedImage, isLeftValid, isRightValid)
        
        
        return outputImage

        
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
        return processedImage


def VideoImageProcessing(image):
    """
    Init
    """
    warpedImage = imageTransform.WarpLaneImage(image)
    processedImage = processing.Process(warpedImage)
    #laneImage = laneId.DetectAndShow(image, warpedImage, processedImage )
    laneImage = laneId.DetectLanes(processedImage )
    unwarpedLaneImage = imageTransform.InverseWarpLaneImage(laneImage)
    #unwarpedLaneImage = imageTransform.InverseWarpLaneImage(laneImage)
    #return image
    return cv2.cvtColor(cv2.addWeighted(image, 1.0, unwarpedLaneImage, 0.3, 0), cv2.COLOR_BGR2RGB) 
     
    
        

def TestImagePipeline():
    # Saving, loading data
    data = Data()
    testImages = data.LoadTestImages()
    imageTransform = ImageTransform()
    processing = ImageProcessing(magnitudeKernelSize=11, angleKernelSize=5)
    laneId = LaneLineIdentification()
    if(imageTransform.isCalibrated):
        for image in testImages:
            laneId.areLanesDetected = False
            warpedImage = imageTransform.WarpLaneImage(image)
            processedImage = processing.Process(warpedImage)
            laneId.DetectAndShow(image, warpedImage,processedImage)


def VideoPipeline(video):
    data =Data()
    if(video == 'project'):
        videoClip = data.LoadProjectVideo()
        outputFile = 'D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/DataStore/ProjectVideo.mp4'
    elif(video == 'challenge'):
        videoClip = data.LoadChallengeVideo()
        outputFile = 'D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/DataStore/ChallengeVideo.mp4'
    elif(video == 'hardchallenge'):
        videoClip = data.LoadHardChallengeVideo()
        outputFile = 'D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/DataStore/HardChallengeVideo.mp4'
        
        
    output = videoClip.fl_image(lambda x:  VideoImageProcessing(cv2.cvtColor(x, cv2.COLOR_RGB2BGR))) 
    output.write_videofile(outputFile, audio=False)


#TestImagePipeline()
processing = ImageProcessing()
imageTransform = ImageTransform()
laneId = LaneLineIdentification()

VideoPipeline('project')   
#VideoPipeline('challenge')
#VideoPipeline('hardchallenge')
print("BigCounter = ", BigCounter) 
    




    

    

