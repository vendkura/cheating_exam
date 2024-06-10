# import cv2
# import numpy as np
# from scipy.spatial import distance as dist


# MODEL_PATH = "yolo-coco"

# MIN_CONF = 0.3
# NMS_THRESH = 0.3

# USE_GPU = False

# MIN_DISTANCE = 250




# def detect_people(frame, net, ln, personIdx=0):
# 	(H, W) = frame.shape[:2]
# 	results = []


# 	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
# 		swapRB=True, crop=False)
# 	net.setInput(blob)
# 	layerOutputs = net.forward(ln)

# 	boxes = []
# 	centroids = []
# 	confidences = []
# 	for output in layerOutputs:

# 		for detection in output:
			
# 			scores = detection[5:]
# 			classID = np.argmax(scores)
# 			confidence = scores[classID]

		
# 			if classID == personIdx and confidence > MIN_CONF:
			
# 				box = detection[0:4] * np.array([W, H, W, H])
# 				(centerX, centerY, width, height) = box.astype("int")

# 				x = int(centerX - (width / 2))
# 				y = int(centerY - (height / 2))

			
# 				boxes.append([x, y, int(width), int(height)])
# 				centroids.append((centerX, centerY))
# 				confidences.append(float(confidence))


# 	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

# 	if len(idxs) > 0:

# 		for i in idxs.flatten():
# 			(x, y) = (boxes[i][0], boxes[i][1])
# 			(w, h) = (boxes[i][2], boxes[i][3])


# 			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
# 			results.append(r)

# 	return results



# face_cascade = cv2.CascadeClassifier(
#     'cascades/haarcascade_profileface.xml')

# faceCascade2 = cv2.CascadeClassifier(
#     'cascades/haarcascade_frontalface_default.xml')

# #network contain weights and configration
# net = cv2.dnn.readNet('9.weights','yolov3-416.cfg')
# #extract object names from coco file
# classes = []
# with open('coco.names' , 'r')as f:
#     classes = f.read().splitlines()
    

# ln = net.getLayerNames()
# ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# LABELS = open("coco.names").read().strip().split("\n")


# camera = cv2.VideoCapture(0)
# while (cv2.waitKey(1) == -1):
#     success, img = camera.read()
#     if success:
#         results = detect_people(img, net, ln,personIdx=LABELS.index("person"))
#         violate = set()
#         if len(results) >= 2:
#             centroids = np.array([r[2] for r in results])
#             D = dist.cdist(centroids, centroids, metric="euclidean")
#             for i in range(0, D.shape[0]):
#                 for j in range(i + 1, D.shape[1]):
#                     if D[i, j] < MIN_DISTANCE:
#                         violate.add(i)
#                         violate.add(j)
				

    
#         for (i, (prob, bbox, centroid)) in enumerate(results):
            
#             (startX, startY, endX, endY) = bbox
#             (cX, cY) = centroid
#             color = (0, 255, 0)
            
#             if i in violate:
#                 color = (0, 0, 255)
            
        
#         cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
#         cv2.circle(img, (cX, cY), 5, color, 1)
        
#         text = "Distance Violations: {}".format(len(violate))
        
#         cv2.putText(img, text, (10, img.shape[0] - 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         height , width , _ = img.shape
#         blob = cv2.dnn.blobFromImage(img , 1/255 , (416,416) , (0,0,0) , swapRB = True , crop = False)
       
#         net.setInput(blob)
        
#         outputLayesNames = net.getUnconnectedOutLayersNames()
#         layerOutputs = net.forward(outputLayesNames)
        
        
#         boxes = []
#         confidences = []
        
#         classes_ids  = []
        
        
#         for output in layerOutputs:
#             for detection in output:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.1 :
#                     centerX = int(detection[0]*width)
#                     centerY = int(detection[1]*height)
#                     w = int(detection[2]*width)
#                     h = int(detection[3]*height)
                    
#                     x = int(centerX - w/2)
#                     y = int(centerY - h/2)
                    
#                     boxes.append([x , y , w ,h ])
#                     confidences.append((float (confidence)))
#                     classes_ids.append(class_id)
                    
        
        
#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
#         font =cv2.FONT_HERSHEY_PLAIN
#         #colors = np.random.uniform(0 , 255 , size = (len(boxes), 3))
        
#         for i  in indexes.flatten():
#             x , y, w ,h  = boxes[i]
#             label = str(classes[classes_ids[i]])
#             confidence  = str(round(confidences[i],2))
#             #color = colors[i]
#             if label == 'cell phone' or label == 'book':
#                 cv2.rectangle(img , (x,y) , (x+w , y+h) , (0,0,255) , 2)
#             else:
#                 cv2.rectangle(img , (x,y) , (x+w , y+h) , (255,255,255) , 2)
#             cv2.putText(img , label+" "+confidence, (x,y+20) , font , 2 , (255,0,0) , 2)
        
        
        
        
        
        
        
        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faceFront = faceCascade2.detectMultiScale(
#             gray, 1.3, 5, minSize=(30, 30))
#         facesL = face_cascade.detectMultiScale(
#             gray, 1.3, 5, minSize=(30, 30))
#         # 1 code to flip horizontaly
#         flipped = cv2.flip(gray , 1)
#         facesR = face_cascade.detectMultiScale(flipped , 1.3  , 5 ,  minSize=(30, 30)  )
#         for (x, y, w, h) in faceFront:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             roi_gray = gray[y:y+h, x:x+w]
#         for (x, y, w, h) in facesL:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#             roi_gray = gray[y:y+h, x:x+w]
#         for (x, y, w, h) in facesR: 
#             cv2.rectangle(img, (x+80,y), ((x+w+50), y+h), (0, 0, 255), 2)
#             roi_gray = gray[y:y+h, x:x+w]
#         cv2.imshow('Cam', img)
#         key = cv2.waitKey(1)
#         if key == 27:
#             break
# camera.release()
# cv2.destroyAllWindows() 

import cv2
import numpy as np
from scipy.spatial import distance as dist
# from playsound import playsound
import datetime
import os
import logging

now = datetime.datetime.now()
# This will give a string like "2022-03-15_14-30-00"
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")

# Set the environment variable for Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"

# journaling the detection
logging.basicConfig(filename=f'./logs/detection_{date_string}.csv', level=logging.INFO, 
                    format='%(asctime)s,%(levelname)s,"%(message)s"')

##---------------------
MODEL_PATH = "yolo-coco"
MIN_CONF = 0.3
NMS_THRESH = 0.3
USE_GPU = False
MIN_DISTANCE = 250



def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > MIN_CONF:
                logging.info(f'Person detected with confidence {confidence}')
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))


    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    if len(idxs) > 0:
        idxs = np.array(idxs)  # Convertir en tableau NumPy si nécessaire
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results


face_cascade = cv2.CascadeClassifier('cascades/haarcascade_profileface.xml')
faceCascade2 = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Network contain weights and configuration
net = cv2.dnn.readNet('yolov3.weights', 'yolov3-416.cfg')
# Extract object names from coco file
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

LABELS = open("coco.names").read().strip().split("\n")

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Use the date string in the output file name
out = cv2.VideoWriter(f'video_recording/output_{date_string}.avi', fourcc, 5.0, (640, 480))



def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


camera = cv2.VideoCapture('/home/asheleyine/github.com/Elnazer/cheating_recognition/video_cheating/video1.mp4')
# camera = cv2.VideoCapture(0)
while (cv2.waitKey(1) == -1):
    success, img = camera.read()
    if success:
        # img = resize_frame(img, 50)
        results = detect_people(img, net, ln, personIdx=LABELS.index("person"))
        violate = set()
        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        logging.info(f'Distance violation: objects {i} and {j} are too close')
                        violate.add(i)
                        violate.add(j)

        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.circle(img, (cX, cY), 5, color, 1)

        # cv2.waitKey(100)
        text = "Distance Violations: {}".format(len(violate))
        cv2.putText(img, text, (10, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
         # Write the frame into the file 'output.avi'
        out.write(img)

        # Play alarm if there is a violation
        # if(len(violate) > 0):
        #     playsound('alarms/alarm2.wav')

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)
        outputLayerNames = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(outputLayerNames)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    centerX = int(detection[0] * width)
                    centerY = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(centerX - w / 2)
                    y = int(centerY - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN  # Définir la police ici

        if len(indexes) > 0:
            indexes = np.array(indexes)  # Convertir en tableau NumPy si nécessaire
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                if label == 'cell phone' or label == 'book':
                    logging.info(f'{label} detected with confidence {confidence}')
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # playsound('alarms/alarm1.wav')
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 0, 0), 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceFront = faceCascade2.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        facesL = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
        flipped = cv2.flip(gray, 1)
        facesR = face_cascade.detectMultiScale(flipped, 1.3, 5, minSize=(30, 30))
        for (x, y, w, h) in faceFront:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
        for (x, y, w, h) in facesL:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
        for (x, y, w, h) in facesR:
            cv2.rectangle(img, (x + 80, y), ((x + w + 50), y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
        cv2.imshow('Cam', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
camera.release()
out.release()
cv2.destroyAllWindows()
