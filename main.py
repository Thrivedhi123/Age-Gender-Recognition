import cv2
import numpy as np
from collections import deque

# Function to detect faces and return bounding boxes
def facebox(faceNet, frame):
    frameH, frameW = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    bbox = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Only consider high-confidence detections
            x1 = int(detections[0, 0, i, 3] * frameW)
            y1 = int(detections[0, 0, i, 4] * frameH)
            x2 = int(detections[0, 0, i, 5] * frameW)
            y2 = int(detections[0, 0, i, 6] * frameH)
            bbox.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw bounding box
    return frame, bbox

# Load face detection model
faceProt = "opencv_face_Detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProt)

# Load age prediction model
ageProt = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProt)

# Load gender prediction model
genderProt = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProt)

# Define age and gender categories
agelist = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderlist = ['Male', 'Female']
mean_values = (78.4263377603, 87.7689143744, 114.89584746)

# Use deque to store last N predictions for smoothing results
N = 10  
age_history = deque(maxlen=N)
gender_history = deque(maxlen=N)

# Open webcam for video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break
    frameN, bbox = facebox(faceNet, frame)  # Detect faces

    for b in bbox:
        face = frameN[b[1]:b[3], b[0]:b[2]]  # Extract face ROI
        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), mean_values, swapRB=False)

        # Predict Gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderlist[genderPred[0].argmax()]
        gender_history.append(gender)

        # Predict Age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = agelist[agePred[0].argmax()]
        age_history.append(age)

        # Take most common prediction from history for stability
        stable_gender = max(set(gender_history), key=gender_history.count)
        stable_age = max(set(age_history), key=age_history.count)

        label = f"{stable_gender}, {stable_age}"

        # Get text size for display
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)
        x, y = b[0], b[1] - 10
        
        # Draw a background rectangle for text
        cv2.rectangle(frameN, (x, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), -1)
        
        # Display the label on screen
        cv2.putText(frameN, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    # Show the video feed with detections
    cv2.imshow("Age - Gender", frameN)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
