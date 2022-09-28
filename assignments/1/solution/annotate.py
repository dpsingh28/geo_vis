import numpy as np
import cv2

points = []
def mouseEvent(event,x,y,flags,params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button clicked. Co-ordinates are ", x,y)
        points.append([x,y]) 

def annotate(image_path):
    global points
    img  = cv2.imread(image_path)
    cv2.imshow("Annotations" , img)
    cv2.setMouseCallback("Annotations" , mouseEvent)
    cv2.waitKey(0)
    points = np.asarray(points)
    return points