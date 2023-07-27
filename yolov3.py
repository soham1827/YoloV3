import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3 # less the number less the repeating bounding boxes

classesfile = 'coco.names.txt'
classNames = []

with open(classesfile,'rt') as f:
	classNames=f.read().rstrip('\n').split('\n')
print(classNames)
print(len(classNames))

modelConfiguration = 'yolov3.cfg.txt'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
	hT, wT, cT = img.shape
	bbox = []
	classIds = []
	confs = []
	for output in outputs:
		for det in output:
			scores = det[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				w,h = int(det[2]*wT), int(det[3]*hT) # converting float to int as they are in percentage
				x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
				bbox.append([x,y,w,h])
				classIds.append(classId)
				confs.append(float(confidence))
	#print(len(bbox))

	indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
	for i in indices:
		i=i[0]
		box = bbox[i]
		x,y,w,h = box[0],box[1],box[2],box[3]
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%'
			, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 1)


while True:
	success, img = cap.read()

	blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
	net.setInput(blob) #blob format as the network understands


	layerNames = net.getLayerNames()
	#print(layerNames)
	outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()] 
	''' we need to get values of each layer and minus 1 as it starts from index value 1'''
	#print(outputNames)

	outputs = net.forward(outputNames) # sends to the network for detection
	#print(len(outputs))
	#print(outputs[0].shape) # generates a matrix
	#print(outputs[1].shape) # generates a matrix

	'''(300,85)- 300 bounding boxes produced by one layer and 
	85 is 80 classes + centre_x + centre_y + width + height + confidence

	so the shape of outputs would give a matrix of 300 rows and 85 columns

	so to detect n create bounding boxes
	we will keep the best confidence result if it clears some threshold in new list else ignore
	so func= findObjects()'''

	findObjects(outputs,img)

	cv2.imshow('Image',img)
	if cv2.waitKey(1) == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()