import cv2
size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('/home/sonali/.local/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/sonali/.local/lib/python3.8/site-packages/cv2/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/home/sonali/.local/lib/python3.8/site-packages/cv2/data/haarcascade_smile.xml') 
eye_glasses=cv2.CascadeClassifier('/home/sonali/.local/lib/python3.8/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml') 

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)
        #Save just the rectangle faces in SubRecFaces
        roi_gray = gray[y:y+h, x:x+w]
        sub_face = im[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20) 
        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(sub_face, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(sub_face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        eye_glass = eye_glasses.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eye_glass:
            cv2.rectangle(sub_face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        FaceFileName = "unknow_" + str(y) + ".jpg"
        #cv2.imwrite(FaceFileName, sub_face)

    # Show the image
    cv2.imshow('BCU Research by Waheed Rafiq (c)',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()