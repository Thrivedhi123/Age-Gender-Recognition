import cv2

def facebox(faceNet,frame):
    frameH = frame.shape[0]
    frameW = frame.shape[1]
    blob= cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox=[]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameW)
            y1=int(detection[0,0,i,4]*frameH)
            x2=int(detection[0,0,i,5]*frameW)
            y2=int(detection[0,0,i,6]*frameH)
            bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

    return frame,bbox



faceProt = "opencv_face_Detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel,faceProt)

ageProt = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProt = "gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel,ageProt)
genderNet = cv2.dnn.readNet(genderModel,genderProt)

agelist = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
genderlist= ['Male','Female']
mean_values = (78.4263377603,87.7689143744,114.89584746)

video = cv2.VideoCapture(0)

while True:
    ret,frame = video.read()
    frameN,bbox = facebox(faceNet,frame)
    
    for b in bbox:
        face= frameN[b[1]:b[3],b[0]:b[2]]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),mean_values,swapRB=False)
        genderNet.setInput(blob)
        genderPred= genderNet.forward()
        gender = genderlist[genderPred[0].argmax()]
        
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = agelist[agePred[0].argmax()]
        
        label = "{},{}".format(gender,age)
        cv2.putText(frameN,label,(b[0],b[1]-10),cv2.FONT_HERSHEY_PLAIN,0.8,(255,255,255),2)
        
    
    cv2.imshow("Age - Gender",frameN)
    k= cv2.waitKey(1)
    
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

