import openface
import cv2
import numpy as np 
import os

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))

img1 = cv2.imread("persons/Shakira/shakira.jpg")
rgbimg1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
bb1 = align.getLargestFaceBoundingBox(rgbimg1)
alignedFace1 = align.align(96, rgbimg1, bb1, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
rep1 = net.forward(alignedFace1)

img2 = cv2.imread("persons/Ben/Ben.jpg")
rgbimg2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
bb2 = align.getLargestFaceBoundingBox(rgbimg2)
alignedFace2 = align.align(96, rgbimg2, bb2, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
rep2 = net.forward(alignedFace2)


print("Representantion 1:")
print(rep1)
print("-----\n")

print("Representantion 2:")
print(rep2)
print("-----\n")

d = rep1 - rep2
distance = np.dot(d,d)
print("Distance:")
print(distance)