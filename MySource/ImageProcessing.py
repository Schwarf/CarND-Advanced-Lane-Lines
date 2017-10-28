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

    def ConvertImageToHSVSpace(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def ConvertHSVImageToGrayColorSpace(self, image):
        image1 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return self.ConvertBGRImageToGrayColorSpace(image1)

    
    def ConvertBackToBGR(self,image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    
    def YellowHSVMask(self,image):
        hsvImage = self.ConvertImageToHSVSpace(image)
        yellowHSVLow = np.array([80,100,100])
        yellowHSVHigh = np.array([110,255,255])
        #yellowHSVLow = np.array([ 255, 255, 0])
        #yellowHSVHigh = np.array([ 255, 255, 204])
        #yellowPixels = cv2.inRange(image, yellowHSVLow, yellowHSVHigh)
        yellowPixels = cv2.inRange(hsvImage, yellowHSVLow, yellowHSVHigh)
        hsvYellowImage = cv2.bitwise_and(hsvImage, hsvImage, mask=yellowPixels)
        bgrYellowImage = cv2.cvtColor(hsvYellowImage, cv2.COLOR_HSV2BGR)
        return bgrYellowImage
    
    def WhiteHSVMask(self,image):
#        hsvImage = self.ConvertImageToHSVSpace(image)
#        whiteHSVLow  = np.array([  0,   0,   200])
#        whiteHSVHigh = np.array([ 255,  80, 255])
        whiteHSVLow = np.array([205, 205, 205])
        whiteHSVHigh = np.array([ 255, 255, 255])
        #whitePixels = cv2.inRange(hsvImage, whiteHSVLow, whiteHSVHigh)
        whitePixels = cv2.inRange(image, whiteHSVLow, whiteHSVHigh)
        return  cv2.bitwise_and(image, image, mask=whitePixels)
    
    def ApplyWhiteAndYellowColorMasks(self,image):
        yellowMask = self.YellowHSVMask(image)
        whiteMask = self.WhiteHSVMask(image)
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

        filteredImage = self.ApplyWhiteAndYellowColorMasks(image)
        #print(filteredImage.shape)
        filteredImage = self.ConvertBGRImageToGrayColorSpace(filteredImage)
        (thresh, processedImage) = cv2.threshold(filteredImage, 1, 255, cv2.THRESH_BINARY)
        
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
        ax2.set_title('Thresholded Magnitude', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
