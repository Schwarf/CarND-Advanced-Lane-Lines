'''
Created on Oct 26, 2017

@author: andre
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob, os

class ImageProcessing:
    def __init__(self, magnitudeKernelSize = 9, angleKernelSize = 3, isBGRImage = True, goToHSVSpace = True, magnitudeThresholdMinimum =50, magnitudeThresholdMaximum =200, 
                 angleThresholdMinimum =0.7, angleThresholdMaximum =1.3, xThresholdMinimum =20, xThresholdMaximum =100, yThresholdMinimum =20, yThresholdMaximum =100):
        self.isBGRImage = isBGRImage
        print("Sobel kernel size for magnitude is ", magnitudeKernelSize)
        print("Sobel kernel size for magnitude is ", angleKernelSize)
        self.sobelMagnitudeKernelSize = magnitudeKernelSize
        self.sobelAngleKernelSize = angleKernelSize
        self.magnitudeThresholdMinimum = magnitudeThresholdMinimum
        self.magnitudeThresholdMaximum = magnitudeThresholdMaximum
        self.angleThresholdMinimum = angleThresholdMinimum
        self.angleThresholdMaximum = angleThresholdMaximum
        self.xThresholdMinimum = xThresholdMinimum
        self.xThresholdMaximum = xThresholdMaximum
        self.yThresholdMinimum = yThresholdMinimum
        self.yThresholdMaximum = yThresholdMaximum

        self.goToHSVSpace = goToHSVSpace

    def ConvertBGRImageToGrayColorSpace(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def ConvertRGBToHSV(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def ConvertHSVImageToGrayColorSpace(self, image):
        image1 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return self.ConvertBGRImageToGrayColorSpace(image1)

    
    def ConvertHSVToBGR(self,image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    def ConvertBGRToLAB(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def ConvertLABToBGR(self,image):
        return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    def ConvertBGRToLUV(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2LUV)

    def ConvertLUVToBGR(self,image):
        return cv2.cvtColor(image, cv2.COLOR_LUV2BGR)
    
    def YellowLABMask(self,image):
        labImage = self.ConvertBGRToLAB(image)
        yellowLABLow = np.array([0,0,155])
        yellowLABHigh = np.array([255,255,200])
        yellowPixels = cv2.inRange(labImage, yellowLABLow, yellowLABHigh)
        labYellowImage = cv2.bitwise_and(labImage, labImage, mask=yellowPixels)
        bgrYellowImage = cv2.cvtColor(labYellowImage, cv2.COLOR_LAB2BGR)
        return bgrYellowImage

    def YellowLUVMask(self,image):
        luvImage = self.ConvertBGRToLUV(image)
        yellowLUVLow = np.array([0,0,155])
        yellowLUVHigh = np.array([255,255,200])
        yellowPixels = cv2.inRange(luvImage, yellowLUVLow, yellowLUVHigh)
        luvYellowImage = cv2.bitwise_and(luvImage, LUVImage, mask=yellowPixels)
        bgrYellowImage = cv2.cvtColor(LUVYellowImage, cv2.COLOR_LUV2BGR)
        return bgrYellowImage

    
    def YellowHSVMask(self,image):
        hsvImage = self.ConvertRGBToHSV(image)
        yellowHSVLow = np.array([0,100,100])
        yellowHSVHigh = np.array([110,255,255])
        
        yellowPixels = cv2.inRange(hsvImage, yellowHSVLow, yellowHSVHigh)
        hsvYellowImage = cv2.bitwise_and(hsvImage, hsvImage, mask=yellowPixels)
        bgrYellowImage = cv2.cvtColor(hsvYellowImage, cv2.COLOR_HSV2BGR)
        return bgrYellowImage
    
    def WhiteBGRMask(self,image):
        whiteHSVLow = np.array([180, 180, 180])
        whiteHSVHigh = np.array([ 255, 255, 255])
        whitePixels = cv2.inRange(image, whiteHSVLow, whiteHSVHigh)
        return  cv2.bitwise_and(image, image, mask=whitePixels)


    def ApplyColorThresholds(self, image):
        firstLUVChannel =cv2.cvtColor(image, cv2.COLOR_BGR2LUV)[:,:,0]
        lowerLThreshold = 225
        upperLThreshold = 255
        lChannelBinary = np.zeros_like(firstLUVChannel)
        lChannelBinary[(firstLUVChannel >= lowerLThreshold) & (firstLUVChannel <= upperLThreshold)] = 1
        
        lastLABChannel =cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:,:,2]
        lowerBThreshold = 155
        upperBThreshold = 200
        bChannelBinary = np.zeros_like(lastLABChannel)
        bChannelBinary[(lastLABChannel >= lowerBThreshold) & (lastLABChannel <= upperBThreshold)] = 1
    
        combinedThresholds = np.zeros_like(lastLABChannel)
        combinedThresholds[(bChannelBinary == 1) | (lChannelBinary == 1)] = 1
        
        
        return combinedThresholds
        
        
    
    def ApplyWhiteAndYellowColorMasks(self,image):
        yellowMask = self.YellowHSVMask(image)
        #yellowMask = self.YellowLABMask(image)
        whiteMask = self.WhiteBGRMask(image)
        filtered = cv2.addWeighted(whiteMask, 1., yellowMask, 1., 0.)
        return filtered
    
    def ApplyGaussianSmoothing(self, image, kernelSize =5):
        return cv2.GaussianBlur(image,(kernelSize, kernelSize),0)

    
    def ApplyOpening(self, image, kernelSize = 5):
        kernel = np.ones((kernelSize,kernelSize),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def ApplyCannyEdgeDetection(self, image, lowerThreshold=50, upperThreshold=150):
        return cv2.Canny(image, lowerThreshold, upperThreshold)


        
    def ApplySobelOperators(self, grayImage, kernelSize):
        sobelX = cv2.Sobel(grayImage, cv2.CV_64F,1,0,ksize = kernelSize)
        sobelY = cv2.Sobel(grayImage, cv2.CV_64F,0,1,ksize = kernelSize)
        return sobelX, sobelY
    
    def AngleGradient(self, grayImage):
        sobelX, sobelY = self.ApplySobelOperators(grayImage, self.sobelAngleKernelSize)
        angle = np.arctan2(np.absolute(sobelY),np.absolute(sobelX))
        gradientImage = np.zeros_like(grayImage)
        gradientImage[(angle > self.angleThresholdMinimum) & (angle < self.angleThresholdMaximum) ] = 1 
        return gradientImage
        

    def MagnitudeGradient(self, grayImage):
        sobelX, sobelY = self.ApplySobelOperators(grayImage, self.sobelMagnitudeKernelSize)
        normXY = np.sqrt(np.square(sobelX) + np.square(sobelY))
        magnitude = np.uint8(255 * normXY/np.max(normXY))
        gradientImage = np.zeros_like(grayImage)
        gradientImage[(magnitude > self.magnitudeThresholdMinimum) & (magnitude < self.magnitudeThresholdMaximum) ] = 1 
        return gradientImage

    def XAndYGradients(self, grayImage):
        sobelX, sobelY = self.ApplySobelOperators(grayImage, self.sobelMagnitudeKernelSize)
        absX = np.absolute(sobelX)
        absY = np.absolute(sobelY)
        X = np.uint8(255 * absX/np.max(absX))
        Y = np.uint8(255 * absY/np.max(absY))

        gradientX = np.zeros_like(sobelX)
        gradientY = np.zeros_like(sobelY)
        gradientX[(X >= self.xThresholdMinimum) & (X <= self.xThresholdMaximum)] =1
        gradientY[(Y >= self.yThresholdMinimum) & (Y <= self.yThresholdMaximum)] =1

        return gradientX, gradientY
    
    def ApplyCorrection(self, image, kernelSize = 3):
        #kernel = np.ones((kernelSize,kernelSize),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernelSize,kernelSize));
        return cv2.dilate(image,kernel,iterations = 1)
    
    
    def Process(self, image):
        
        #filteredImage = self.ApplyWhiteAndYellowColorMasks(image)
        #processedImage = self.ConvertBGRImageToGrayColorSpace(filteredImage)
        #(thresh, processedImage) = cv2.threshold(processedImage, 1, 255, cv2.THRESH_BINARY)
        processedImage = self.ApplyColorThresholds(image)
        #print(filteredImage.shape)
        
        #
        
        #print(grayImage)
        #smoothedImage = self.ApplyGaussianSmoothing(grayImage)
#        edgeImage = self.ApplyCannyEdgeDetection(smoothedImage)
        #magnitudeImage = self.MagnitudeGradient(filteredImage)
        #angleImage = self.AngleGradient(filteredImage)
        #magnitudeImage = self.MagnitudeGradient(grayImage)
        #angleImage = self.AngleGradient(smoothedImage)
#        xGradientImage, yGradientImage =  self.XAndYGradients(grayImage)
        #magnitudeImage = self.ApplyCorrection(magnitudeImage)
        #angleImage = self.ApplyCorrection(angleImage)
        #processedImage = np.zeros_like(image)
        #processedImage[( (grayImage ==1))] =1 
        #processedImage[((magnitudeImage ==1) | (angleImage ==1))] =1
        #processedImage = self.DefineTetragonROIAndApplyToImage(processedImage, 0.4)
        
        return processedImage

 
    def ShowGradientThresholding(self):
        grayImage = self.ConvertBGRImageToGrayColorSpace(filteredImage)
        angleImage = self.AngleGradient(grayImage)
        magnitudeImage = self.MagnitudeGradient(grayImage)
        processedImage[((magnitudeImage ==1) | (angleImage ==1))] =1
        (thresh, processedImage) = cv2.threshold(processedImage, 1, 255, cv2.THRESH_BINARY)
        #xGradientImage, yGradientImage =  self.XAndYGradients(grayImage)
        f, (p1, p2) = plt.subplots(1, 2, figsize=(12, 9))
        p1.imshow(image)
        p1.set_title('Original image', fontsize=20)
        p2.imshow(processedImage)
        p2.set_title('Apply thresholding for angle gradient and magnitude gradient', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    
    def ShowProcessedImage(self, image):
        scalingFactor = 0.5
        processedImage = self.Process(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        #grayImage = cv2.resize(grayImage,(0,0), fx=scalingFactor, fy=scalingFactor)
        #processedImage = cv2.resize(processedImage,(0,0), fx=scalingFactor, fy=scalingFactor)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(processedImage, cmap='gray')
        ax2.set_title('Processed Image', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
