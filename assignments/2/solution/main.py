from matplotlib import lines
import numpy as np 
import cv2
import argparse
from solvers import get_cam_matrix

def project_and_normalize_pts(pts):
    # print("Incoming points: \n",pts)
    pts_new = np.empty((pts.shape[0] , pts.shape[1] +1))
    pts_new[: , 0:pts.shape[1]] = pts
    pts_new[:,-1] = -1
    pts_norm  = np.linalg.norm(pts_new,axis=1)
    pts_new = pts_new / np.reshape(pts_norm , (-1,1))
    # print("Outgoing points:,\n",pts_new)
    return pts_new

# def make_2d_lines(pts1 , pts2):
#     assert pts1.shape==pts2.shape , "Shape of both point sets should be same, to make a line"
#     lines = np.reshape(np.array([] , dtype=float) , (0,3))

#     for i in range(pts1.shape[0]):
#         new_line = np.cross(pts1[i,:] , pts2[i,:])
#         lines = np.vstack((lines , new_line))
#     return lines


if __name__== '__main__':
    parser = argparse.ArgumentParser(description="main file for assignment 2 of 16822 Geometry-Based Vision")
    parser.add_argument('--type' ,required=True)

    args = parser.parse_args()

    points  = np.loadtxt('../assignment2/data/q1/bunny.txt')
    pts2 = points[:,0:2]
    pts3 = points[:,2:]
    pts2 = project_and_normalize_pts(pts2)
    pts3 = project_and_normalize_pts(pts3)
    P = get_cam_matrix(pts2 , pts3)
    print("Camera Matrix: \n" , P)

    surface_pts = np.load('../assignment2/data/q1/bunny_pts.npy')
    surface_pts = project_and_normalize_pts(surface_pts)

    img_pts = (P@surface_pts.T).T
    print(img_pts)
    img_pts = img_pts / np.reshape(img_pts[:,-1] , (-1,1))
    img_pts = np.array(-img_pts[: , 0:2] , dtype=int)
    print(img_pts)

    bunny_img = cv2.imread('../assignment2/data/q1/bunny.jpeg')
    for pt in img_pts:
        cv2.drawMarker(bunny_img, (pt[0], pt[1]),(0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=1)
    
    cv2.imshow('Bunny' , bunny_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    cv2.imwrite( './submissions/bunny_pts.jpg' , bunny_img)


#Bounding Box
    new_bunny = cv2.imread('../assignment2/data/q1/bunny.jpeg')
    lines = np.load('../assignment2/data/q1/bunny_bd.npy')
    pt_set1 = lines[:,0:3]
    pt_set2 = lines[:,3:6]
    pt_set1 = project_and_normalize_pts(pt_set1)
    pt_set2 = project_and_normalize_pts(pt_set2)
    img_pt_set1 = (P@pt_set1.T).T
    img_pt_set2 = (P@pt_set2.T).T
    img_pt_set1 = -img_pt_set1/np.reshape(img_pt_set1[:,-1] , (-1,1))
    img_pt_set2 = -img_pt_set2/np.reshape(img_pt_set2[:,-1] , (-1,1))
    img_pt_set1 = np.array(img_pt_set1[:,0:2] , dtype=int)
    img_pt_set2 = np.array(img_pt_set2[:,0:2] , dtype=int)

    assert img_pt_set1.shape == img_pt_set2.shape , "Shape of point sets should be same, to make the bounding box"
    for i in range(img_pt_set1.shape[0]):
        line_img = cv2.line(new_bunny , img_pt_set1[i,:] , img_pt_set2[i,:] , (255,0,0) , 5)
    
    cv2.imshow("Cuboid" , line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite( './submissions/bunny_bbox.jpg' , line_img)
    