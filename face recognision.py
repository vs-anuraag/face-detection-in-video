import cv2 as cv

# Load the cascade
face_cascade = cv.CascadeClassifier('haar_face.xml')

# To capture video from webcam. 
capture = cv.VideoCapture(0)
#can type a path instead of 0 in '' if you want to feed a specific video

while True:
    # Read the frame
    _, video = capture.read()

    # Convert to grayscale
    gray = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(video, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv.imshow('video', video)

    # Stop if escape key is pressed
    p = cv.waitKey(30) & 0xff
    if p==ord('d'):
        break
        
# Release the VideoCapture object
capture.release()