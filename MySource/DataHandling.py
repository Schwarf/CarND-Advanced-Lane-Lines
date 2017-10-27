'''
Created on Oct 26, 2017

@author: andre
'''
import glob, os
import cv2

class Data():
    def __init__(self):
        self.chessBoardImagePath =  "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/camera_cal/*.jpg"
        self.testImagePath =  "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/test_images/*.jpg"

    def LoadChessBoardImages(self):
        calibrationDataPath =  self.chessBoardImagePath
        jpgFiles = [file for file in glob.glob(calibrationDataPath)]
        images = []
        for filePath in jpgFiles:
            image = cv2.imread(filePath)
            images.append(image)
        return images

    def LoadTestImages(self):
        testDataPath =  self.testImagePath
        jpgFiles = [file for file in glob.glob(testDataPath)]
        images = []
        for filePath in jpgFiles:
            image = cv2.imread(filePath)
            images.append(image)
        return images
    