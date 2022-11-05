import numpy as np 
import cv2
import argparse
import math
import annotate
from solvers import eight_pt, seven_pt

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

def getF(object):
    correspondences = np.load('../assignment3/data/q1a/'+object+'/'+ object +'_corresp_raw.npz')
    pts1 = correspondences['pts1']
    pts2 = correspondences['pts2']
    assert pts1.shape == pts2.shape , "Issue in original data"
    T1 = normalize_pts(pts1)
    T2 = normalize_pts(pts2)
    pts1 = (T1 @ project_pts(pts1).T).T
    pts2 = (T2 @ project_pts(pts2).T).T

    return (T2.T @ eight_pt(pts1 , pts2) @ T1)

def epilines(object1, object2 , F_object):
    object1_pt = annotate.annotate(object1)
    object1_pt = project_pts(object1_pt)
    object2_line = (F_object @ object1_pt.T).T
    line_img = object2.copy()
    for line in object2_line:
        x0, y0 = map(int, [0, -line[2] / line[1] ])
        x1, y1 = map(int, [object2.shape[1], -(line[2] + line[0] * object2.shape[1]) / line[1] ])
        line_img = cv2.line(line_img , (x0 , y0) , (x1 , y1) , (255,0,0) , 2)
    show_img('epipolar line' , line_img)

def F_ransac_roots():
    pass


def ransac(object , points , error_thresh , n_iters):
    print(points,'-point algorithm on ',object,'-----Number of iterations= ',n_iters)
    raw_correspondences = np.load('../assignment3/data/q1b/'+object+'/'+object+'_corresp_raw.npz')
    pts1 = raw_correspondences['pts1']
    pts2 = raw_correspondences['pts2']
    F_int = None
    F_best = None
    F_eight = None
    F_seven = None
    history_inliers = 0
    idx_best = None
    for i in range(n_iters):
        # print("Iteration: ",i)
        rand_idx = np.random.randint(0 , pts1.shape[0]-1 , points)
        rand_pts1 = pts1[rand_idx,:]
        rand_pts2 = pts2[rand_idx,:]
        T1 = normalize_pts(rand_pts1)
        T2 = normalize_pts(rand_pts2)
        rand_pts1 = (T1 @ project_pts(rand_pts1).T).T
        rand_pts2 = (T2 @ project_pts(rand_pts2).T).T
        if(points == 8):
            F_eight = T2.T @ eight_pt(rand_pts1 , rand_pts2) @ T1
        elif(points==7):
            roots , F_coeff , F_const =  seven_pt(rand_pts1 , rand_pts2) 
        else:
            raise RuntimeError("points varible can only be 7 or 8")
        
        def ransac_inliers(F_temp):
            errors = project_pts(pts2) * (F_temp @ project_pts(pts1).T).T
            errors = np.abs(np.sum(errors , axis=1))
            check = errors < error_thresh
            idx = (np.where(check == True)[0])
            n_inliers = idx.shape[0]
            return n_inliers
        
        if points ==8:
            num_inliers = ransac_inliers(F_eight)
            F_int = F_eight
        elif points == 7:
            real_part_roots = roots.real
            imag_part_roots = roots.imag
            num_real = np.bincount(roots.imag == 0)[-1]
            if(num_real == 1):
                idx = np.where(imag_part_roots == 0)[0]
                F_seven = real_part_roots[idx]*F_coeff + F_const
                num_inliers = ransac_inliers(F_seven)
            elif(num_real==3):
                Fs = np.zeros((3,3,3))
                sevenp_inlners = np.zeros(3)
                Fs[0,:,:] = real_part_roots[0]*F_coeff + F_const
                Fs[1,:,:] = real_part_roots[1]*F_coeff + F_const
                Fs[2,:,:] = real_part_roots[2]*F_coeff + F_const
                for i in range(3):
                    sevenp_inlners[i] = ransac_inliers(Fs[i,:,:])
                bestF_idx = np.argmax(sevenp_inlners)
                F_seven = Fs[bestF_idx , :,:]
                num_inliers = sevenp_inlners[bestF_idx]
            else:
                raise RuntimeError("[Ransac] Error: number of real roots can be only 1 or 3")
            F_int = F_seven

        if num_inliers > history_inliers:
            history_inliers = num_inliers
            F_best = F_int
            # idx_best = idx
    
    print("Ratio of inliers: ",history_inliers/pts1.shape[0])
    F_best = F_best / F_best[-1,-1]
    print("F_ransac:" , F_best)
    object1 = cv2.imread('../assignment3/data/q1b/'+object+'/image_1.jpg')
    object2 = cv2.imread('../assignment3/data/q1b/'+object+'/image_2.jpg')
    epilines(object1 , object2 , F_best)

    # new_pt1 = pts1[idx_best , :]
    # new_pt2 = pts2[idx_best , :]
    # T1 = normalize_pts(new_pt1)
    # T2 = normalize_pts(new_pt2)
    # new_pt1 = (T1 @ project_pts(new_pt1).T).T
    # new_pt2 = (T2 @ project_pts(new_pt2).T).T
    # F_new = T2.T @ eight_pt(new_pt1 , new_pt2) @ T1
    # F_new = F_new / F_new[-1,-1]
    # print("F_New\n:", F_new)

def F_from_roots(roots, F_coeff , F_const , pts1 , pts2):
    real_part_roots = roots.real
    imag_part_roots = roots.imag
    num_real = np.bincount(roots.imag == 0)[-1]

    if(num_real == 1):
        idx = np.where(imag_part_roots == 0)
        # print("index is: ", idx)
    elif(num_real==3):
        F0 = real_part_roots[0] * F_coeff + F_const
        F1 = real_part_roots[1] * F_coeff + F_const
        F2 = real_part_roots[2] * F_coeff + F_const
        mag = np.zeros(3)
        mag[0] = np.abs(np.trace(pts2 @ F0 @ pts1.T))
        mag[1] = np.abs(np.trace(pts2 @ F1 @ pts1.T))
        mag[2] = np.abs(np.trace(pts2 @ F2 @ pts1.T))
        idx = np.argmin(mag)
        # print("index is: ", idx)
    else:
        raise RuntimeError("Error: number of real roots can be only 1 or 3")

    F_mat = real_part_roots[idx]*F_coeff + F_const
    F_mat = F_mat/F_mat[-1,-1]
    return F_mat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= "Main file for HW3 of 16822")
    parser.add_argument('--type' , required=True, choices=['1A1' , '1A2' , '1B' , '2' , '3' , '4' , '5'])
    args = parser.parse_args()

    if(args.type == '1A1'):
        F_teddy = getF('teddy')
        F_chair = getF('chair')
        print("Fundamental matrix for teddy:\n",F_teddy)
        print("Fundamental matrix for chair:\n",F_chair)
        np.save('./new_data/F_teddy.npy' , F_teddy)
        np.save('./new_data/F_chair.npy' , F_chair)
        object = 'chair'
        object1 = cv2.imread('../assignment3/data/q1a/'+object+'/image_1.jpg')
        object2 = cv2.imread('../assignment3/data/q1a/'+object+'/image_2.jpg')
        epilines(object1 , object2 , F_chair)
        object = 'teddy'
        object1 = cv2.imread('../assignment3/data/q1a/'+object+'/image_1.jpg')
        object2 = cv2.imread('../assignment3/data/q1a/'+object+'/image_2.jpg')
        epilines(object1 , object2 , F_teddy)
    
    elif(args.type == '1A2'):
        F_teddy = np.load('./new_data/F_teddy.npy')
        F_chair = np.load('./new_data/F_chair.npy')
        teddy_intrinsics = np.load('../assignment3/data/q1a/teddy/intrinsic_matrices_teddy.npz')
        chair_intrinsics = np.load('../assignment3/data/q1a/chair/intrinsic_matrices_chair.npz')
        teddy_K1 = teddy_intrinsics['K1']
        teddy_K2 = teddy_intrinsics['K2']
        chair_K1 = chair_intrinsics['K1']
        chair_K2 = chair_intrinsics['K2']
        E_teddy = teddy_K2.T @ F_teddy @ teddy_K1
        E_chair = chair_K2.T @ F_chair @ chair_K1
        print("Essential matrix for teddy\n",E_teddy)
        print("Essential matrix for chair\n",E_chair)

    elif(args.type == '1B'):
        toybus_correspondences = np.load('../assignment3/data/q1b/toybus/toybus_7_point_corresp.npz')
        toytrain_correspondences = np.load('../assignment3/data/q1b/toytrain/toytrain_7_point_corresp.npz')
        
        toybus_pts1 = toybus_correspondences['pts1']
        toybus_pts2 = toybus_correspondences['pts2']
        toytrain_pts1 = toytrain_correspondences['pts1']
        toytrain_pts2 = toytrain_correspondences['pts2']

        T1_toybus = normalize_pts(toybus_pts1)
        T2_toybus = normalize_pts(toybus_pts2)
        toybus_pts1 = (T1_toybus @ project_pts(toybus_pts1).T).T
        toybus_pts2 = (T2_toybus @ project_pts(toybus_pts2).T).T
        roots_toybus , F1_toybus , F2_toybus = seven_pt(toybus_pts1 , toybus_pts2)
        F_toybus = F_from_roots(roots_toybus , F1_toybus , F2_toybus , toybus_pts1 , toybus_pts2)
        F_toybus = T2_toybus.T @ F_toybus @ T1_toybus
        print("F_toybus:\n" , F_toybus)
        toybus_im1 = cv2.imread("../assignment3/data/q1b/toybus/image_1.jpg")
        toybus_im2 = cv2.imread("../assignment3/data/q1b/toybus/image_2.jpg")
        epilines(toybus_im1 , toybus_im2 , F_toybus)

        T1_toytrain = normalize_pts(toytrain_pts1)
        T2_toytrain = normalize_pts(toytrain_pts2)
        toytrain_pts1 = (T1_toytrain @ project_pts(toytrain_pts1).T).T
        toytrain_pts2 = (T2_toytrain @ project_pts(toytrain_pts2).T).T
        roots_toytrain , F1_toytrain , F2_toytrain = seven_pt(toytrain_pts1 , toytrain_pts2)
        F_toytrain = F_from_roots(roots_toytrain , F1_toytrain , F2_toytrain , toytrain_pts1 , toytrain_pts2)
        F_toytrain = T2_toytrain.T @ F_toytrain @ T1_toytrain
        print("F_toytrain:\n", F_toytrain)
        toytrain_im1 = cv2.imread("../assignment3/data/q1b/toytrain/image_1.jpg")
        toytrain_im2 = cv2.imread("../assignment3/data/q1b/toytrain/image_2.jpg")
        epilines(toytrain_im1 , toytrain_im2 , F_toytrain)

    elif(args.type == '2'):
        # ransac('toybus' , 8 , 0.0065 , 10000)
        # ransac('toytrain' , 8 , 0.008 , 10000)
        ransac('toybus' , 7 , 500 , 5000)


        # for pts in pts1[idx_best, :]:
        #     cv2.drawMarker(chair1 , (int(pts[0]) , int(pts[1])) , markerType=cv2.MARKER_DIAMOND, color=(0,0,255) , thickness=5 , markerSize=-5)
        # for pts in pts2[idx_best, :]:
        #     cv2.drawMarker(chair2 , (int(pts[0]) , int(pts[1])) , markerType=cv2.MARKER_DIAMOND, color=(0,0,255) , thickness=5 , markerSize=-5)
        # cv2.imshow('im1' , chair1)
        # show_img('im2' , chair2)