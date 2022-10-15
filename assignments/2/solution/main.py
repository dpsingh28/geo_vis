import numpy as np 
import cv2
import argparse
from solvers import get_cam_matrix, get_w3
from annotate import annotate

def project_and_normalize_pts(pts):
    # print("Incoming points: \n",pts)
    pts_new = np.empty((pts.shape[0] , pts.shape[1] +1))
    pts_new[: , 0:pts.shape[1]] = pts
    pts_new[:,-1] = 1
    pts_norm  = np.linalg.norm(pts_new,axis=1)
    pts_new = pts_new / np.reshape(pts_norm , (-1,1))
    # print("Outgoing points:,\n",pts_new)
    return pts_new

def draw_2dline(pt_set1 , pt_set2 , image):
    new_img = image.copy()
    pt_set1 = project_and_normalize_pts(pt_set1)
    pt_set2 = project_and_normalize_pts(pt_set2)
    img_pt_set1 = (P@pt_set1.T).T
    img_pt_set2 = (P@pt_set2.T).T
    img_pt_set1 = img_pt_set1/np.reshape(img_pt_set1[:,-1] , (-1,1))
    img_pt_set2 = img_pt_set2/np.reshape(img_pt_set2[:,-1] , (-1,1))
    img_pt_set1 = np.array(img_pt_set1[:,0:2] , dtype=int)
    img_pt_set2 = np.array(img_pt_set2[:,0:2] , dtype=int)

    assert img_pt_set1.shape == img_pt_set2.shape , "Shape of point sets should be same, to make the bounding box"
    for i in range(img_pt_set1.shape[0]):
        line_img = cv2.line(new_img , img_pt_set1[i,:] , img_pt_set2[i,:] , (255,0,0) , 5)

    return line_img

# def make_2d_lines(pts1 , pts2):
#     assert pts1.shape==pts2.shape , "Shape of both point sets should be same, to make a line"
#     lines = np.reshape(np.array([] , dtype=float) , (0,3))

#     for i in range(pts1.shape[0]):
#         new_line = np.cross(pts1[i,:] , pts2[i,:])
#         lines = np.vstack((lines , new_line))
#     return lines

def get_projective_line(p1 , p2):
    P = np.vstack((np.reshape(p1 , (1,-1)),np.reshape(p2,(1,-1))))
    P = project_and_normalize_pts(P)
    p1 = P[0,:]
    p2 = P[1,:]
    line = np.expand_dims(np.cross(p1,p2) ,axis=0)
    # print("Line equation: ",line)
    return line

def flip_color(color):
    if color == (255,0,0):
        return (0,255,0)
    elif color == (0,255,0):
        return (0,0,255)
    else:
        return (255,0,0)

def make_lines(img, points):
    new_img = img.copy()
    color = (255,0,0)
    lines = np.reshape(np.array([] , dtype=int) , (0,3))
    # img = cv2.imread(image_path)
    if(len(points)%2 !=0):
        raise Exception("Number of points selected not even")

    for i in range(int(len(points) /2)):
        p1  = points[2*i,:]
        p2 = points[2*i +1,:]
        if (i%2 ==0):
            color = flip_color(color)

        line_img = cv2.line(new_img , p1, p2 , color , 2)
        lines = np.append(lines , get_projective_line(p1,p2) ,axis =0)

    cv2.imshow("Line" , line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('processed_images/annotations/' + out_name + '.jpg' , line_img)

    return lines

if __name__== '__main__':
    parser = argparse.ArgumentParser(description="main file for assignment 2 of 16822 Geometry-Based Vision")
    parser.add_argument('--type' ,required=True, choices=['1a' , '1b' , '2a' , '2b'])

    args = parser.parse_args()

    if(args.type == '1a'):
        print("\033[1;31mBunny \033[1;0m")
        points  = np.loadtxt('../assignment2/data/q1/bunny.txt')
        pts2 = points[:,0:2]
        pts3 = points[:,2:]
        pts2 = project_and_normalize_pts(pts2)
        pts3 = project_and_normalize_pts(pts3)
        P = get_cam_matrix(pts2 , pts3)
        print("\033[1;33mCamera Matrix:\n" , P, "\033[1;0m")

        surface_pts = np.load('../assignment2/data/q1/bunny_pts.npy')
        surface_pts = project_and_normalize_pts(surface_pts)

        img_pts = (P@surface_pts.T).T
        # print(img_pts)
        img_pts = img_pts / np.reshape(img_pts[:,-1] , (-1,1))
        img_pts = np.array(img_pts[: , 0:2] , dtype=int)
        # print(img_pts)

        bunny_img = cv2.imread('../assignment2/data/q1/bunny.jpeg')
        for pt in img_pts:
            cv2.drawMarker(bunny_img, (pt[0], pt[1]),(0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=1)
        
        cv2.imshow('Bunny' , bunny_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite( './submissions/bunny_pts.jpg' , bunny_img)


    #Bounding Box
        new_bunny = cv2.imread('../assignment2/data/q1/bunny.jpeg')
        lines = np.load('../assignment2/data/q1/bunny_bd.npy')
        pt_set1 = lines[:,0:3]
        pt_set2 = lines[:,3:6]
        line_img = draw_2dline(pt_set1 , pt_set2 , new_bunny)

        cv2.imshow("Cuboid" , line_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite( './submissions/bunny_bbox.jpg' , line_img)
    
    elif(args.type == '1b'):
        print("\033[1;31mCuboid \033[1;0m")
        cuboid = cv2.imread('./new_data/cuboid.jpg')

        pts_3d_orig = np.array([[0,0,0],[8.75,0,0],[0,12.25,0],[0,0,5.25],[8.75,0,5.25],[0,12.25,5.25]])
        print("\033[1;32mOriginal Points in 3D \n" , pts_3d_orig, "\033[1;0m")
        #Getting origin point
        pts_2d = annotate(cuboid)
        pts_3d = project_and_normalize_pts(pts_3d_orig)
        pts_2d = project_and_normalize_pts(pts_2d)
        P = get_cam_matrix(pts_2d , pts_3d)
        print("\033[1;33mCamera Matrix:\n" , P, "\033[1;0m")

        pt_set_3d_1 = pts_3d_orig[0:3,:]
        pt_set_3d_2 = pts_3d_orig[3:6,:]
        pt_set_3d_1 = np.vstack((pt_set_3d_1 , np.delete(pts_3d_orig ,(1,2,4,5) , 0)))
        pt_set_3d_1 = np.vstack((pt_set_3d_1 , np.delete(pts_3d_orig ,(1,2,4,5) , 0)))
        pt_set_3d_2 = np.vstack((pt_set_3d_2 , np.delete(pts_3d_orig ,(0,2,3,5) , 0)))
        pt_set_3d_2 = np.vstack((pt_set_3d_2 , np.delete(pts_3d_orig ,(0,1,3,4) , 0)))

        line_img = draw_2dline(pt_set_3d_1 , pt_set_3d_2 , cuboid)
        cv2.imshow('Cuboid' , line_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('./submissions/cuboid_lines.jpg' , line_img)
    
    elif(args.type == '2a'):
        build_img = cv2.imread('../assignment2/data/q2a.png')

        line_pts = annotate(build_img)
        lines = make_lines(build_img , line_pts)
        vanish_pts = np.reshape(np.array([] , dtype=float) ,(0,3))

        for i in range(int(lines.shape[0]/2)):
            new_pt =np.cross(lines[2*i,:],lines[2*i+1,:])
            vanish_pts = np.vstack((vanish_pts , np.reshape(new_pt, (1,-1))))
        w = get_w3(vanish_pts)
        print("IAC value:\n", w)
        L = np.linalg.cholesky(w)
        K = np.linalg.inv(L.T)
        print("Instrinsics matrix: \n", K)

        print_img = build_img.copy()
        print_vanish_pts = np.array(vanish_pts[:,0:2] / np.reshape(vanish_pts[:,2] , (-1,1)) , dtype=int)
        max_pt = np.max(np.absolute(print_vanish_pts))
        print_img = cv2.copyMakeBorder(print_img , max_pt , max_pt , max_pt , max_pt , cv2.BORDER_CONSTANT , None, value=(255,255,255))
        print_vanish_pts = print_vanish_pts + max_pt
        for pt in print_vanish_pts:
            cv2.drawMarker(print_img, (pt[0], pt[1]),(0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=20, thickness=20)
        
        cv2.imshow('Vanishing Points' , print_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print_vanish_pts = np.vstack((print_vanish_pts , print_vanish_pts))
        vanish_line_img = print_img.copy()
        for i in range(int(len(print_vanish_pts) /2)):
            p1  = print_vanish_pts[2*i,:]
            p2 = print_vanish_pts[2*i +1,:]
            cv2.line(vanish_line_img, p1, p2 , (255,0,0) , 10)

        cv2.imshow('Vanishing Points with Line' , vanish_line_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('./submissions/building_vanish_pts.jpg' , vanish_line_img)
