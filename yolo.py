import cv2
import numpy as np


cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)
# print(len(classNames))

modelCon = "E:/projects/yolo v3/yolov3.cfg"
modelWeight = "E:/projects/yolo v3/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelCon, modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    ht,wt,ct = img.shape
    bbox = []
    confs = []
    classIds = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores) # returns index
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wt), int(det[3]*ht) # det[2] gives width percentage of box wrt to image
                x,y = int((det[0]*wt)-w/2),int((det[1]*ht)-h/2) # center coordinates x,y
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold) # Can HAve multiple boxes, this function keeps largest box with most confidence, return index from list
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {confs[i]*100}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)





while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)

    findObjects(outputs,img)
    cv2.imshow('Image', img)
    cv2.waitKey(1)