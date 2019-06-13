import openface
import cv2
import numpy as np 
import os
from sklearn import svm

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))


def alignImg(img):
    rgbimg = img.getRGB()
    bb = align.getLargestFaceBoundingBox(rgbimg)
    alignedFace = align.align(96, rgbimg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return alignedFace
        

def main():
    persons = ["igor", "miguel", "rafael", "renan", "ricardo", "rudyer", "vinicius"]
    persons_rep = []

    # Reading the persons that already know
    for person in persons:
        imgs = list(openface.data.iterImgs("raw/"+ person))
        rep = []
        for i,img in enumerate(imgs,start=1):
            aligned = alignImg(img)
            if aligned is None:
                line = person + " " + str(i) + "  nao reconhecida"
                print(line)
            else:
                rep.append(net.forward(aligned))
        persons_rep.append(rep)
    
    # Reading the unknown person
    img = openface.data.Image("unkown", "2.jpg", "raw/desconhecido").getRGB()
    img_aligned = alignImg(img)
    if img_aligned is None:
        print("nao reconhecido")
        exit()
    else:
        img_rep = net.forward(img_aligned)

    # Fiting into the SVC classifier
    samples = []
    labels = []

    for i,person in enumerate(persons_rep, start=1):
        labels.append(i)
        samples.append(person)
    

    
    
    

    
        

main()