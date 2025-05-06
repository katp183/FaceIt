import numpy as np
import tensorflow as tf
import cv2
import os

trainPath = "C:\Users\kevin\archive\train"
testPath = "C:\Users\kevin\archive\test"

folderList = os.listdir(trainPath)
folderList.sort()

print(folderList)

