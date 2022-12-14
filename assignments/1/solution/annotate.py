import numpy as np
import cv2

points = []
def mouseEvent(event,x,y,flags,params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button clicked")
        points.append([x,y]) 

def annotate(img):
    global points
    points=[]
    # img  = cv2.imread(image_path)
    cv2.imshow("Annotations" , img)
    cv2.setMouseCallback("Annotations" , mouseEvent)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    points = np.asarray(points)
    return points