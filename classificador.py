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

#search for openface.data.iterImgs

def alignImg(img):
    rgbimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbimg)
    alignedFace = align.align(96, rgbimg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return alignedFace

def main():
    persons = ["igor", "miguel", "rafael", "renan", "ricardo", "rudyer", "vinicius"]
    persons_rep = []

    for person in persons:
        rep = []
        print(person)
        for i in range(1,6):
            img = cv2.imread("raw/" + person + "/" + str(i) + ".jpg")
            aligned = alignImg(img)
            print(i)
            aux = net.forward(aligned)
            rep.append(aux)
        persons_rep.append(rep)

    for i in range(0,6):
        print(person(i))
        print(persons_rep(i))

main()