#   CONFIGURATION

import openface
import cv2
import numpy as np 
import os
from sklearn import svm

fileDir = os.path.dirname(os.path.realpath(__file__))
frameWorkDir = os.path.join(fileDir, "..")
modelDir = os.path.join(frameWorkDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))

#persons = ["igor", "miguel", "renan", "vinicius"]

#   ORGANIZE THE DATASET

datasetDir = os.path.join(fileDir, "know")
imgs = list(openface.data.iterImgs(datasetDir))    

#   RECOGNIZE FACES FROM DATASET

#   know
reps = {}
for img in imgs:
    rgbimg = img.getRGB()
    className = img.cls
    if className not in reps:
        reps[className] = []
    imgName = img.name
    face = align.getLargestFaceBoundingBox(rgbimg)
    if face is None:
        print("Can't recognize " + imgName + " from " + className + ".")
    else:
        alignedFace = align.align(96, rgbimg, face, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        reps[className].append(net.forward(alignedFace))

#   unknow
unknowimgs = list(openface.data.iterImgs(os.path.join(fileDir,"unknown")))
unknowReps = []
for img in unknowimgs:
    rgbimg = img.getRGB()
    face = align.getLargestFaceBoundingBox(rgbimg)
    if face is None:
        print("Can't recognize " + imgName + " from unknown person.")
    else:
        alignedFace = align.align(96, rgbimg, face, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        unknowReps.append(net.forward(alignedFace))

#   FIT INTO A MULTI-CLASS AND A ONE-CLASS SVM CLASSIFIER

samples = []
labels = []
for key in reps:
    labels += [key] * len(reps[key])
    for rep in reps[key]:
        samples.append(rep)

multiClassCLF = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)
multiClassCLF.fit(samples,labels)
oneClassCLF = svm.OneClassSVM(nu=0.9, kernel="rbf", gamma='scale')
oneClassCLF.fit(samples)

#   DETECT NOVELTY ===============> PROBLEMS!!!
for rep in unknowReps:
    print(oneClassCLF.predict(rep.reshape(1,-1)))
    print(multiClassCLF.predict(rep.reshape(1,-1)))

#   NEW FACE



#   OLD FACE