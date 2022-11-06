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
    cv2.imshow("Point Annotations" , img)
    cv2.setMouseCallback("Point Annotations" , mouseEvent)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    points = np.asarray(points)

    pt_img = img.copy()
    for i in range(len(points)):
        pt_img = cv2.circle(pt_img, tuple(points[i,:]), radius=5, color=(255, 0, 0), thickness=-1)
        # pt_img = cv2.putText(pt_img , str(i) , tuple([points[i,0] +2 , points[i,1]]) , fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=1 , color=(0,255,0) , thickness=2)
    cv2.imshow("Annotated Points" , pt_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./submissions/chair_pts_8p.jpg' , pt_img)
    return points