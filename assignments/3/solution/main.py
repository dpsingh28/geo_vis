import numpy as np 
import cv2
import argparse
import math
import annotate
from solvers import eight_pt

# np.random.seed(5)

def show_img(window_name , image):
    cv2.imshow(window_name , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def project_pts(pts):
    pts_new = np.empty((pts.shape[0] , pts.shape[1] +1))
    pts_new[: , 0:pts.shape[1]] = pts
    pts_new[:,-1] = 1
    return pts_new

def normalize_pts(pts):
    center = np.sum(pts , axis=0) / pts.shape[0]
    x0 , y0 = center
    d_avg = pts - center
    d_avg = np.linalg.norm(d_avg , axis=1)
    d_avg = np.sum(d_avg) / len(d_avg)
    sv = math.sqrt(2) / d_avg
    T_mat = np.array([[sv , 0 , -sv*x0] , [0 , sv , -sv*y0] , [0 , 0 , 1]])
    return T_mat

def q1a1(object):
    correspondences = np.load('../assignment3/data/q1a/'+object+'/'+ object +'_corresp_raw.npz')
    pts1 = correspondences['pts1']
    pts2 = correspondences['pts2']
    assert pts1.shape == pts2.shape , "Issue in original data"
    # idx = np.random.randint(low=0 , high=pts1.shape[0] , size=8, dtype=int)
    # idx = [0,1,2,3,4,5,6,7]
    # pts1 = pts1[idx , :]
    # pts2 = pts2[idx , :]
    T1 = normalize_pts(pts1)
    T2 = normalize_pts(pts2)
    pts1 = (T1 @ project_pts(pts1).T).T
    pts2 = (T2 @ project_pts(pts2).T).T

    return (T2.T @ eight_pt(pts1 , pts2) @ T1)

def q1a1_epiline(object , F_object):
    object1 = cv2.imread('../assignment3/data/q1a/'+object+'/image_1.jpg')
    object2 = cv2.imread('../assignment3/data/q1a/'+object+'/image_2.jpg')
    object1_pt = annotate.annotate(object1)
    object1_pt = project_pts(object1_pt)
    object2_line = np.squeeze(F_object @ np.reshape(object1_pt , (-1,1)))
    x0, y0 = map(int, [0, -object2_line[2] / object2_line[1] ])
    x1, y1 = map(int, 
                    [object2.shape[1], -(object2_line[2] + object2_line[0] * object2.shape[1]) / object2_line[1] ])
    line_img = cv2.line(object2.copy() , (x0 , y0) , (x1 , y1) , (255,0,0) , 5)
    show_img('epipolar line' , line_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Main file for HW3 of 16822")
    parser.add_argument('--type' , required=True, choices=['1A1' , '1A2' , '1B' , '2' , '3' , '4' , '5'])
    args = parser.parse_args()

    if(args.type == '1A1'):
        F_teddy = q1a1('teddy')
        F_chair = q1a1('chair')
        print(F_teddy)
        print(F_chair)
        q1a1_epiline('chair' , F_chair)
        q1a1_epiline('teddy' , F_teddy)





        # for item in arr:
        #     item = np.array(item , dtype=int)
        #     cv2.drawMarker(chair2, (item[0], item[1]),(0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=-2, thickness=2, line_type=cv2.LINE_AA)