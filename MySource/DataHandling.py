'''
Created on Oct 26, 2017

@author: andre
'''
import glob, os
import cv2
import pickle
from moviepy.editor import VideoFileClip

class Data():
    def __init__(self):
        self.chessBoardImagePath =  "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/camera_cal/*.jpg"
        self.testImagePath =  "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/test_images/*.jpg"
        self.cameraVariablesPath =  "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/DataStore/CameraVariables.p"
        self.perspectiveTransformVariablesPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/DataStore/PerspectiveTransformVariables.p"
        self.projectVideoPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/project_video.mp4"
        self.challengeVideoPath = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/challenge_video.mp4"
        self.hardChallengeVideoPath ="D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project4/CarND-Advanced-Lane-Lines/harder_challenge_video.mp4"
        self.CalibrationMatrixString = "CalibrationMatrix"
        self.DistortionCoefficientString = "DistortionCoefficients"
        self.WarpMatrixString = "WarpMatrix"
        self.InverseWarpMatrixString = "InverseWarpMatrix"
        self.cameraFileExists = os.path.isfile(self.cameraVariablesPath)
        self.perspectiveTransformFileExists = os.path.isfile(self.perspectiveTransformVariablesPath)


        
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
    
    def SaveCameraCalibrationVariables(self, cameraMatrix, distortionCoefficients):
        calibrationPickle = {}
        calibrationPickle[self.CalibrationMatrixString] = cameraMatrix
        calibrationPickle[self.DistortionCoefficientString] = distortionCoefficients
        pickle.dump( calibrationPickle, open( self.cameraVariablesPath, "wb" ) )

    
    def SavePerspectiveTransfromVariables(self, warpMatrix, inverseWarpMatrix):
        calibrationPickle = {}
        calibrationPickle[self.WarpMatrixString] = warpMatrix
        calibrationPickle[self.InverseWarpMatrixString] = inverseWarpMatrix
        pickle.dump( calibrationPickle, open( self.perspectiveTransformVariablesPath, "wb" ) )
    
    def LoadCameraCalibrationVariables(self):
        calibrationPickle = pickle.load(open( self.cameraVariablesPath, "rb" ) )
        cameraMatrix = calibrationPickle[self.CalibrationMatrixString]
        distortionCoefficients = calibrationPickle[self.DistortionCoefficientString]
        return cameraMatrix, distortionCoefficients

    def LoadPerspectiveTransfomrVariables(self):
        calibrationPickle = pickle.load(open( self.perspectiveTransformVariablesPath, "rb" ) )
        warpMatrix = calibrationPickle[self.WarpMatrixString]
        inverseWarpMatrix = calibrationPickle[self.InverseWarpMatrixString]
        return warpMatrix, inverseWarpMatrix
    
        
    def LoadProjectVideo(self):
        return VideoFileClip(self.projectVideoPath)
    
    def LoadChallengeVideo(self):
        return VideoFileClip(self.challengeVideoPath)
    
    def LoadHardChallengeVideo(self):
        return VideoFileClip(self.hardChallengeVideoPath)