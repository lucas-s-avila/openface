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
    if rgbimg is None:
        return None
    bb = align.getLargestFaceBoundingBox(rgbimg)
    alignedFace = align.align(96, rgbimg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return alignedFace
        

def main():
    '''
    persons = ["miguel", "rudyer"]
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
    '''
    # Reading the unknown person
    imgs = list(openface.data.iterImgs("raw/vinicius"))
    img_aligned = alignImg(imgs[4])
    if img_aligned is None:
        print("nao reconhecido")
        exit()
    else:
        img_rep = net.forward(img_aligned)


    imgs = list(openface.data.iterImgs("raw/igor"))
    rep1 = net.forward(alignImg(imgs[0]))
    rep1 += rep12

    imgs = list(openface.data.iterImgs("raw/rudyer"))
    rep2 = net.forward(alignImg(imgs[0]))
    
    imgs = list(openface.data.iterImgs("raw/miguel"))
    rep3 = net.forward(alignImg(imgs[0]))

    # Fiting into the SVC classifier
    samples = [rep1, rep2, rep3]
    labels = ["igor", "rudyer", "miguel"]
    
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(samples, labels)

    # predicting the unknown face
    print(clf.predict(img_rep.reshape(1,-1)))
    print(clf.support_vectors_)


    '''
    To do:
        classificar todas as imagens dos rostos conhecidos dentro do svm;
        ao ler um rosto desconhecido, achar as distâncias do rosto pros SVM's;
        se menor ou maior que tal distância (a definir), aplicar clf.predict().
    '''   
        

main()