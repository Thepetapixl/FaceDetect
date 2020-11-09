from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

#load the photograph

pixels = imread(r'/Users/admin/VScode/FaceDetect/MTCNN/OpenCV/test2.jpg')

# load the pretrained model 

classifier = CascadeClassifier(r'/Users/admin/VScode/FaceDetect/MTCNN/OpenCV/haarcascade_frontalface_default.xml')

#perform face detection 

bboxes = classifier.detectMultiScale(pixels, 1.05, 3)

#print bounding box for each detected face

for box in bboxes:
    #extract
    x,y,width, height = box
    x2, y2 = x + width, y + height
    #draw rectangle over the pixels
    rectangle(pixels,(x,y),(x2,y2),(0,0,255),1)

# show the image 
imshow('face dectection', pixels)
# keep the window open until we press a key

waitKey(0)

destroyAllWindows()