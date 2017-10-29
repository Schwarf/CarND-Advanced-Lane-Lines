'''
Created on Oct 26, 2017

@author: andre
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from DataHandling import Data

class PerspectiveTransform:
    def __init__(self, yRegionRatio = 0.4, xTetragonTopWidth =100, xTetragonBottomWidthReduction=200):
        self.objectPoints = []
        self.imagePoints = []
        self.pointsPerRow = 9
        self.pointsPerColumn = 6
        self.cameraMatrix = None
        self.distortionCoefficients = None
        self.warpMatrix = None
        self.inverseWarpMatrix = None
        self.data = Data()
        self.isCalibrated = False
        self.yRegionRatio = yRegionRatio
        self.xTetragonTopWidth = xTetragonTopWidth
        self.xTetragonBottomWidthReduction   = xTetragonTopWidth
        self.Calibrate()

        
    def GenerateCalibrationData(self,image, pointsPerRow, pointsPerColumn):
        grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        objectPointsInitialization = np.zeros((pointsPerColumn*pointsPerRow,3), np.float32)
        objectPointsInitialization[:,:2] = np.mgrid[0:pointsPerRow,0:pointsPerColumn].T.reshape(-1,2)
        ret, chessBoardCorners = cv2.findChessboardCorners(grayImage, (pointsPerRow,pointsPerColumn), None)
        if not ret :
            raise Exception("Finding chess board corners in method 'GenerateCalibrationData' failed")
        
        self.imagePoints.append(chessBoardCorners)
        self.objectPoints.append(objectPointsInitialization)
        return grayImage



    def Calibrate(self):
        if self.data.cameraFileExists:
            self.cameraMatrix, self.distortionCoefficients = self.data.LoadCameraCalibrationVariables()
            self.isCalibrated = True
            return self.isCalibrated
        
        images = self.data.LoadChessBoardImages()
        pointsPerRow = self.pointsPerRow
        pointsPerColumn = self.pointsPerColumn
        for image in images:
            print("Calibrating ...")
            grayImage= self.GenerateCalibrationData(image, pointsPerRow, pointsPerColumn)
        self.isCalibrated, self.cameraMatrix, self.distortionCoefficients, cameraRotationVector, cameraTranslationVector = cv2.calibrateCamera(self.objectPoints, self.imagePoints, grayImage.shape[::-1], None, None)    
        
        self.data.SaveCameraCalibrationVariables(self.cameraMatrix, self.distortionCoefficients)
        
        return self.isCalibrated
    
    
    
    def UndistortImage(self, image):
        calibratedImage = cv2.undistort(image, self.cameraMatrix, self.distortionCoefficients) 
        return calibratedImage
    
    def WarpChessBoardImage(self, image):
        if not self.isCalibrated:
            self.Calibrate()
        
        undistortedImage = self.UndistortImage(image)
        grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, chessBoardCorners = cv2.findChessboardCorners(grayImage, (self.pointsPerRow,self.pointsPerColumn), None)
        offset = 100
        if not ret :
            raise Exception("Finding chess board corners in method 'WarpChessBoardImage' failed")
        
        sourcePoints = np.float32([chessBoardCorners[0], chessBoardCorners[self.pointsPerRow-1], chessBoardCorners[-1], chessBoardCorners[-self.pointsPerRow]])
        imageWidth = grayImage.shape[1]
        imageHeight = grayImage.shape[0]
        #print("Width and Height (WarpChessBoardImage)", imageWidth, imageHeight)
        destinationPoints = np.float32([[offset,offset], [imageWidth-offset, offset], [imageWidth-offset, imageHeight-offset], [offset, imageHeight-offset]])
        warpMatrix = cv2.getPerspectiveTransform(sourcePoints, destinationPoints)
        warpedImage = cv2.warpPerspective(undistortedImage, warpMatrix, (imageWidth, imageHeight))
        return warpedImage 
    

    
    def GetSourcePoints(self, height, width, bottomWidthLeftCorrection, bottomWidthRightCorrection, topLeftWidthCorrection, topRightWidthCorrection, heightCorrection):
        leftBottom = bottomWidthLeftCorrection
        rightBottom = bottomWidthRightCorrection
        leftTop = topLeftWidthCorrection
        rightTop = topRightWidthCorrection
        bottom = height - 45
        top = heightCorrection
        #print('SourcePoints', np.float32([[(leftBottom, bottom), (leftTop, top), (rightTop, top), (rightBottom, bottom)]]))
        return np.float32([[(leftBottom, bottom), (leftTop, top), (rightTop, top), (rightBottom, bottom)]])

    
    def GetDestinationPoints(self, height, width, widthLeftCorrection, widthRightCorrection):
        left = widthLeftCorrection
        right = widthRightCorrection
        top =0 
        bottom = height
        return np.float32([[(left, bottom), (left, top), (right, top), (right, bottom)]])
        
    def DefineWarpTransformation(self, image, printPolyLines):
        undistortedImage = self.UndistortImage(image)
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        
        destinationPoints = self.GetDestinationPoints(imageHeight, imageWidth, widthLeftCorrection=384, widthRightCorrection =896)
        sourcePoints = self.GetSourcePoints(imageHeight, imageWidth, bottomWidthLeftCorrection=384, bottomWidthRightCorrection= 896,  
                                            topLeftWidthCorrection=581, topRightWidthCorrection = 699, heightCorrection=477)
        
        if(printPolyLines):
            undistortedImage = cv2.polylines(undistortedImage,[sourcePoints.astype(np.int32)],True,(0,0,255), 4)
            image = cv2.polylines(image,[sourcePoints.astype(np.int32)],True,(0,0,255), 4)
        
        self.warpMatrix = cv2.getPerspectiveTransform(sourcePoints, destinationPoints)
        self.inverseWarpMatrix = cv2.getPerspectiveTransform(destinationPoints, sourcePoints)
        self.data.SavePerspectiveTransfromVariables(self.warpMatrix, self.inverseWarpMatrix)
    
        
    def WarpLaneImage(self,image, printPolyLines = False):
        if not self.isCalibrated:
            self.Calibrate()
        if self.data.perspectiveTransformFileExists:
            self.warpMatrix, self.inverseWarpMatrix = self.data.LoadPerspectiveTransfomrVariables()
        else:
            self.DefineWarpTransformation(image, printPolyLines)

        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        undistortedImage = self.UndistortImage(image)
        warpedImage = cv2.warpPerspective(undistortedImage, self.warpMatrix, (imageWidth, imageHeight))
        return warpedImage
    
    def InverseWarpLaneImage(self, image):
        imageWidth = image.shape[1]
        imageHeight = image.shape[0]
        inverseWarpedImage = cv2.warpPerspective(image, self.inverseWarpMatrix, (imageWidth, imageHeight))
        return inverseWarpedImage
    
    def ShowTransformResult(self, image):   
        #calibratedImage = self.WarpChessBoardImage(image) 
        
        #grayImage = cv2.resize(grayImage,(0,0), fx=scalingFactor, fy=scalingFactor)
        #processedImage = cv2.resize(processedImage,(0,0), fx=scalingFactor, fy=scalingFactor)
        calibratedImage = self.WarpLaneImage(image,False)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(calibratedImage, cmap='gray')
        ax2.set_title('Thresholded Magnitude', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
