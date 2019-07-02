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


def disMed(rep, vectors, n):
    aux=0
    for i in range(3):
        distances=[]
        for j in range(n[i]):
            d = rep - vectors[j+aux]
            d = np.dot(d,d)
            distances.append(d)
        aux+=n[i]
        print(str(np.mean(distances)) + " - " + str(np.median(distances)) + " - " + str(np.std(distances)))
        

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
    imgs = list(openface.data.iterImgs("raw/desconhecido"))
    img = imgs[0]
    print(img.name)
    img_aligned = alignImg(img)
    if img_aligned is None:
        print("nao reconhecido")
        exit()
    else:
        img_rep = net.forward(img_aligned)


    imgs = list(openface.data.iterImgs("raw/igor"))
    rep1 = []
    for i,img in enumerate(imgs):
        aligned = alignImg(img)
        if aligned is None:
            line = "Igor " + str(i) + "  nao reconhecida"
            print(line)
        else:
            rep1.append(net.forward(aligned))
    
    imgs = list(openface.data.iterImgs("raw/rudyer"))
    rep2 = []
    for i,img in enumerate(imgs):
        aligned = alignImg(img)
        if aligned is None:
            line = "Rudyer " + str(i) + "  nao reconhecida"
            print(line)
        else:
            rep2.append(net.forward(aligned))
    
    imgs = list(openface.data.iterImgs("raw/miguel"))
    rep3 = []
    for i,img in enumerate(imgs):
        aligned = alignImg(img)
        if aligned is None:
            line = "Miguel " + str(i) + "  nao reconhecida"
            print(line)
        else:
            rep3.append(net.forward(aligned))

    # Fiting into the SVC classifier
    samples = rep1 + rep2 + rep3
    labels1 = ["igor"] * len(rep1)
    labels2 = ["rudyer"] * len(rep2)
    labels3 = ["miguel"] * len(rep3)
    labels = labels1 + labels2 + labels3
    
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)
    clf.fit(samples, labels)

    # predicting the unknown face
    print(clf.predict(img_rep.reshape(1,-1)))

    disMed(img_rep, clf.support_vectors_, clf.n_support_)

    #disMed(img_rep, clf.support_vectors_, clf.n_support_[1])

    #disMed(img_rep, clf.support_vectors_, clf.n_support_[2])


    '''
    To do:
        classificar todas as imagens dos rostos conhecidos dentro do svm;
        ao ler um rosto desconhecido, achar as distancias do rosto pros SVM's;
        se menor ou maior que tal distancia (a definir), aplicar clf.predict().
    '''   
        

main()