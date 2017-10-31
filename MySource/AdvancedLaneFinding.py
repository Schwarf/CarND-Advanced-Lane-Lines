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
    def __init__(self, imageAreaFactor = 0.99, numberOfSlidingWindows = 9, windowWidth =100, video='project'):
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
        self.leftLaneCurvature = None
        self.rightLaneCurvature = None
        self.relativeCarPosition = None
        if(video == 'project'):
            self.normalizedMeanDeviationThreshold = 0.5
        else:
            self.normalizedMeanDeviationThreshold = 0.08
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
                
                normalizedMeanSquareDeviation = np.sqrt(np.mean(np.divide( (result -averageValue)**2,averageValue**2)))
                if(normalizedMeanSquareDeviation < self.normalizedMeanDeviationThreshold):
                    average.append(result)
                else:
                    print('ALERT ALERT')
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
        
        self.CurvatureDetermination(yValues, leftLaneFitX, rightLaneFitX)
        self.CalculateRelativeCarPosition(imageWidth, leftLaneFitX, rightLaneFitX)
        
        leftPoints = np.array([np.transpose(np.vstack([leftLaneFitX, yValues]))])
        rightPoints = np.array([np.flipud(np.transpose(np.vstack([rightLaneFitX, yValues])))])
        allPoints = np.hstack((leftPoints, rightPoints))
        
        cv2.fillPoly(outputImage, np.int_([allPoints]), (0,255, 0))
        return outputImage
    


    def CalculateRelativeCarPosition(self, imageWidth, leftLaneFitX, rightLaneFitX):
        xMeterPerPixel = 3.7/700.
        imageHorizontalCenter = np.int(imageWidth/2)
        laneHorizontalCenter = (rightLaneFitX[-1] - leftLaneFitX[-1]) / 2.0 + leftLaneFitX[-1]
        self.relativeCarPosition = (imageHorizontalCenter - laneHorizontalCenter) * xMeterPerPixel

    
    def CurvatureDetermination(self, yValues, leftLaneFitX, rightLaneFitX):
        yMeterPerPixel = 30./720.
        xMeterPerPixel = 3.7/700.

        worldCoordinateLeftX = xMeterPerPixel*leftLaneFitX
        worldCoordinateLeftY = yMeterPerPixel*yValues
        worldCoordinateRightX = xMeterPerPixel*rightLaneFitX
        worldCoordinateRightY = worldCoordinateLeftY 

        leftLaneFitInWorldCoordinates = self.UpdateQuadraticFit(worldCoordinateLeftX, worldCoordinateLeftY)  
        rightLaneFitInWorldCoordinates = self.UpdateQuadraticFit(worldCoordinateRightX, worldCoordinateRightY)
        maxYValue =np.max(yValues)*yMeterPerPixel
        self.leftLaneCurvature = ((1 + (2*leftLaneFitInWorldCoordinates[0]*maxYValue + leftLaneFitInWorldCoordinates[1])**2)**1.5) / np.absolute(2*leftLaneFitInWorldCoordinates[0])
        self.rightLaneCurvature = ((1 + (2*rightLaneFitInWorldCoordinates[0]*maxYValue + rightLaneFitInWorldCoordinates[1])**2)**1.5) / np.absolute(2*rightLaneFitInWorldCoordinates[0])
        # in kilometers
        self.leftLaneCurvature /=1000.0
        self.rightLaneCurvature /=1000.0
         
    
    
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

        
        

        
            
    def DetectAndShow(self, image, warpedImage, processedImage, imageTransform):
        global original
        laneImage = self.DetectLanes(processedImage)
        
        warpedImage = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2RGB)
        unwarpedLaneImage = imageTransform.InverseWarpLaneImage(laneImage)
        
        finalImage = cv2.cvtColor(cv2.addWeighted(image, 1.0, unwarpedLaneImage, 0.3, 0), cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #histogramMidpoint, histogram = self.CalculateHistogram(processedImage, 0, processedImage.shape[0])

        f, plots = plt.subplots(3, 2, figsize=(12, 15))
        plots[0,0].imshow(image)
        plots[0,0].set_title('Original ', fontsize=20)
        plots[0,1].imshow(warpedImage)
        plots[0,1].set_title('Image warped ', fontsize=20)
        plots[1,0].imshow(processedImage, cmap='gray')
        plots[1,0].set_title('Image warped and processed ', fontsize=20)
        plots[1,1].imshow(laneImage)
        plots[1,1].set_title('Identified lane', fontsize=20)
        plots[2,0].imshow(unwarpedLaneImage)
        plots[2,0].set_title('Unwarped identified lane', fontsize=20)
        plots[2,1].imshow(finalImage)
        plots[2,1].set_title('Original image with identified lane', fontsize=20)
        
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        self.areLanesDetected = False
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
    overlayImage = cv2.cvtColor(cv2.addWeighted(image, 1.0, unwarpedLaneImage, 0.3, 0), cv2.COLOR_BGR2RGB)
    cv2.putText(overlayImage, "Left lane curvature radius: {0:.2f} km ".format( laneId.leftLaneCurvature), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    cv2.putText(overlayImage, "Right lane curvature radius: {0:.2f} km ".format( laneId.rightLaneCurvature), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    cv2.putText(overlayImage, "Relative car position: {0:.2f} m ".format( laneId.relativeCarPosition), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    #unwarpedLaneImage = imageTransform.InverseWarpLaneImage(laneImage)
    #return image
    return  overlayImage
     
    
        

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


#video = 'project'
video = 'challenge'

processing = ImageProcessing()
imageTransform = ImageTransform()
laneId = LaneLineIdentification(video=video)


VideoPipeline(video)

    




    

    

