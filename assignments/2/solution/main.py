#!/usr/bin/env python3
import math
import time
import numpy as np
import open3d as o3d
import cv2
import argparse
from solvers import get_cam_matrix, get_K3,  get_H, get_K5
from annotate import annotate
import annotations
from shapely.geometry import Polygon, Point

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

def show_img(window_name , image):
    cv2.imshow(window_name , image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def fill_poly(poly_img , poly_pts , clr):
    new_img = poly_img.copy()
    cv2.fillPoly(new_img , pts=[poly_pts] , color=clr)
    show_img("Polygon" , new_img)

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

    show_img("Line" , line_img)
    # cv2.imwrite('processed_images/annotations/' + out_name + '.jpg' , line_img)
    return lines

def get_angle(n1 , n2):
    rad = np.arccos(np.dot(n1,n2)/(np.linalg.norm(n1)*np.linalg.norm(n2)))
    deg = (rad*180)/math.pi
    return deg

def angles_and_normals_from_pts(sq_img , sq_pts , intrinsics):
    pts_matrix = np.reshape(np.array([] , dtype=int) , (0,2))
    for i in range(sq_pts.shape[0]):
        sq_pt1 = sq_pts[i,:,:]
        sq_temp1 = sq_pt1.copy()
        sq_temp1[1:, :] = np.roll(sq_temp1[1:, :] , 1, axis=0) 
        pts_matrix = np.array(np.vstack((pts_matrix , sq_pt1 , sq_temp1))  , dtype=int)

    lines = make_lines(sq_img , pts_matrix)
    v=[]
    for i in range(int(len(lines)/2)):
        v.append(np.cross(lines[2*i,:] , lines[2*i +1,:] ))
    v= np.asarray(v)
    print("Vanishing points:\n", v)

    dir_vec = (np.linalg.inv(intrinsics) @ v.T).T
    normals= []
    for i in range(int(len(dir_vec)/2)):
        normals.append(np.cross(dir_vec[2*i,:] , dir_vec[2*i+1,:] ))
    normals = np.asarray(normals)
    print("Normals:\n",normals)

    print("1 and 2:", get_angle(normals[0,:] , normals[1,:]))
    print("2 and 3:", get_angle(normals[1,:] , normals[2,:]))
    print("3 and 1:", get_angle(normals[2,:] , normals[0,:]))
    return normals

def get_vanishpts(vanish_lines):
    vanishpts = np.reshape(np.array([] , dtype=float) ,(0,3))
    for i in range(int(vanish_lines.shape[0]/2)):
        new_pt =np.cross(vanish_lines[2*i,:],vanish_lines[2*i+1,:])
        vanishpts = np.vstack((vanishpts , np.reshape(new_pt, (1,-1))))
    return vanishpts

############################################################################### Main Function ###############################################################################

if __name__== '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="\033[1;31mGeneral Description:\033[0;0m main file for assignment 2 of 16822 Geometry-Based Vision\
                                                                                                \n\033[1;31mFor Q3:\033[0;0m first_run variable is required to run q3\
                                                                                                \n\t-give False if intrinsics, normals and plane points npy files exist in submissions directory\
                                                                                                \n\t-give True if this is the first run. All values will be saved in submissions directory")
    parser.add_argument('--type' ,required=True, choices=['1a' , '1b' , '2a' , '2b' , '2c' , '3a'])
    parser.add_argument('--first_run' ,required=False, type=str , choices=['True', 'False'])
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
        img_pts = img_pts / np.reshape(img_pts[:,-1] , (-1,1))
        img_pts = np.array(img_pts[: , 0:2] , dtype=int)

        bunny_img = cv2.imread('../assignment2/data/q1/bunny.jpeg')
        for pt in img_pts:
            cv2.drawMarker(bunny_img, (pt[0], pt[1]),(0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=1)
        show_img('Bunny' , bunny_img)
        save_path = './submissions/bunny_pts.jpg'
        cv2.imwrite(save_path , bunny_img)
        print("Output file saved to ",save_path)


        #Bounding Box
        new_bunny = cv2.imread('../assignment2/data/q1/bunny.jpeg')
        lines = np.load('../assignment2/data/q1/bunny_bd.npy')
        pt_set1 = lines[:,0:3]
        pt_set2 = lines[:,3:6]
        line_img = draw_2dline(pt_set1 , pt_set2 , new_bunny)
        save_path = './submissions/bunny_bbox.jpg'
        show_img("Cuboid" , line_img)
        cv2.imwrite( save_path , line_img)
        print("Output file saved to ",save_path)
    

    elif(args.type == '1b'):
        print("\033[1;31mCuboid \033[1;0m")
        cuboid = cv2.imread('./new_data/cuboid.jpg')

        pts_3d_orig = np.array([[0,0,0],[8.75,0,0],[0,12.25,0],[0,0,5.25],[8.75,0,5.25],[0,12.25,5.25]])
        print("\033[1;32mOriginal Points in 3D \n" , pts_3d_orig, "\033[1;0m")
        
        #Getting 3D points
        print("\033[1;33mAdd annotations for co-ordinate system. Points to be chosen in order shown in the assignment handout\033[0m")
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
        show_img('Cuboid' , line_img)
        save_path = './submissions/cuboid_lines.jpg'
        cv2.imwrite(save_path , line_img)
        print("Output file saved to ",save_path)

    
    elif(args.type == '2a'):
        build_img = cv2.imread('../assignment2/data/q2a.png')
        new_line_pts = np.load('./data/q2/q2a.npy')
        
        line_pts = np.vstack((new_line_pts[0,:,:] , new_line_pts[1,:,:] , new_line_pts[2,:,:]))
        line_pts = np.reshape(line_pts , (-1,2))
        lines = make_lines(build_img , line_pts)
        vanish_pts = get_vanishpts(lines)
        K_2a = get_K3(vanish_pts)
        print("Instrinsics matrix: \n", K_2a)

        #Visualization of vanishing points
        print_img = build_img.copy()
        print_vanish_pts = np.array(vanish_pts[:,0:2] / np.reshape(vanish_pts[:,2] , (-1,1)) , dtype=int)
        max_pt = np.max(np.absolute(print_vanish_pts))
        print_img = cv2.copyMakeBorder(print_img , max_pt , max_pt , max_pt , max_pt , cv2.BORDER_CONSTANT , None, value=(255,255,255))
        print_vanish_pts = print_vanish_pts + max_pt
        for pt in print_vanish_pts:
            cv2.drawMarker(print_img, (pt[0], pt[1]),(0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=20, thickness=20)
        
        show_img('Vanishing Points' , print_img)
        print_vanish_pts = np.vstack((print_vanish_pts , print_vanish_pts))
        vanish_line_img = print_img.copy()
        for i in range(int(len(print_vanish_pts) /2)):
            p1  = print_vanish_pts[2*i,:]
            p2 = print_vanish_pts[2*i +1,:]
            cv2.line(vanish_line_img, p1, p2 , (255,0,0) , 10)
        
        save_path = './submissions/building_vanish_pts.jpg'
        show_img('Vanishing Points with Line' , vanish_line_img)
        cv2.imwrite(save_path , vanish_line_img)
        print("Output image saved to ",save_path)


    elif(args.type == '2b'):
        square_img = cv2.imread('../assignment2/data/q2b.png')
        square_pts = np.load('../assignment2/data/q2/q2b.npy')
        annotations.vis_annnotations_q2b()

        square_pts1 = square_pts[0,:,:]
        square_pts2 = square_pts[1,:,:]
        square_pts3 = square_pts[2,:,:]

        H1 = get_H(project_and_normalize_pts(square_pts1) , 1)
        H2 = get_H(project_and_normalize_pts(square_pts2) , 1)
        H3 = get_H(project_and_normalize_pts(square_pts3) , 1)
        
        K_2b = get_K5(H1,H2,H3)
        print("Instrinsics Matrix: \n", K_2b)
        _ = angles_and_normals_from_pts(square_img , square_pts , K_2b)
        
    
    elif(args.type == '2c'):
        wh1 = 32.5/14.5
        wh2 = 14/9
        wh3 = 14/9.25
        square_img = cv2.imread('./new_data/rectangle.jpg')
        print("\033[1;31mAdd first square points in a clockwise manner,\033[33m starting from top left\033[0m")
        square_pts1 = annotate(square_img)
        fill_poly(square_img , square_pts1 , (255,255,255))

        print("\033[1;31mAdd second square points in a clockwise manner,\033[33m starting from top left\033[0m")
        square_pts2 = annotate(square_img)
        fill_poly(square_img , square_pts2 , (255,0,0))

        print("\033[1;31mAdd third square points in a clockwise manner,\033[33m starting from top left\033[0m")
        square_pts3 = annotate(square_img)
        fill_poly(square_img , square_pts3 , (0,255,0))

        H1 = get_H(project_and_normalize_pts(square_pts1) , wh1)
        H2 = get_H(project_and_normalize_pts(square_pts2) , wh2)
        H3 = get_H(project_and_normalize_pts(square_pts3) , wh3)

        K_2c = get_K5(H1,H2,H3)
        print("Instrinsics Matrix: \n", K_2c)
        square_pts_stack = np.zeros((3,4,2) , dtype=int)
        square_pts_stack[0,:,:] = square_pts1
        square_pts_stack[1,:,:] = square_pts2
        square_pts_stack[2,:,:] = square_pts3
        print(square_pts_stack)
        _ = angles_and_normals_from_pts(square_img , square_pts_stack, K_2c)


    elif(args.type == '3a'):
        building_img = cv2.imread('./data/q3.png')
        plane_pts = np.load('./data/q3/q3.npy')
        print("Plane points are:\n",plane_pts)
        annotations.vis_annotations_q3()

        polygon1 = Polygon(plane_pts[0,:,:])
        polygon2 = Polygon(plane_pts[1,:,:])
        polygon3 = Polygon(plane_pts[2,:,:])
        polygon4 = Polygon(plane_pts[3,:,:])
        polygon5 = Polygon(plane_pts[4,:,:])
        
        xs = np.arange(0, building_img.shape[1] , 1)
        ys = np.arange(0, building_img.shape[0] , 1)
        xv , yv = np.meshgrid(xs,ys)
        coords = np.vstack([yv.ravel() , xv.ravel()])
        
        def get_polygon_pts(polygon):
            pts_p = np.reshape(np.array([] , dtype=int) , (2,0))
            for pt in coords.T:
                if(polygon.contains(Point(pt))):
                    print("Calculating points in different planes : true")
                    pts_p = np.hstack((pts_p , np.reshape(pt , (2,-1))))
                else:
                    print("Calculating points in different planes : false")
            return pts_p


        assert args.first_run!=None , "Please input first_run variable value"

        if(args.first_run == 'True'):
            print("Add annotations for calculating vanishing points(3 sets of parallel lines)")
            line_pts = annotate(building_img)
            lines = make_lines(building_img , line_pts)
            vanish_pts = get_vanishpts(lines)
            K_3a = get_K3(vanish_pts)
            normals_3a = angles_and_normals_from_pts(building_img , plane_pts , K_3a)

            pts_p1 = get_polygon_pts(polygon1)
            pts_p2 = get_polygon_pts(polygon2)
            pts_p3 = get_polygon_pts(polygon3)
            pts_p4 = get_polygon_pts(polygon4)
            pts_p5 = get_polygon_pts(polygon5)

            print("Intrinsics are:\n", K_3a)
            print("Normals are:\n", normals_3a)
            print("Shape of ptsp1: ", pts_p1.shape)
            print("Shape of ptsp2: ", pts_p2.shape)
            print("Shape of ptsp3: ", pts_p3.shape)
            print("Shape of ptsp4: ", pts_p4.shape)
            print("Shape of ptsp5: ", pts_p5.shape)

            np.save('./submissions/ptsp1.npy' , pts_p1)
            np.save('./submissions/ptsp2.npy' , pts_p2)
            np.save('./submissions/ptsp3.npy' , pts_p3)
            np.save('./submissions/ptsp4.npy' , pts_p4)
            np.save('./submissions/ptsp5.npy' , pts_p5)
            np.save('./submissions/intrinsics_3a.npy' , K_3a)
            np.save('./submissions/normals_3a.npy' , normals_3a)
        else:
            print("\nLoading saved data")
            normals_3a = np.load('./submissions/normals_3a.npy')
            K_3a = np.load('./submissions/intrinsics_3a.npy')
            pts_p1 = np.load('./submissions/ptsp1.npy')
            pts_p2 = np.load('./submissions/ptsp2.npy')
            pts_p3 = np.load('./submissions/ptsp3.npy')
            pts_p4 = np.load('./submissions/ptsp4.npy')
            pts_p5 = np.load('./submissions/ptsp5.npy')

            print("Intrinsics are:\n", K_3a)
            print("Normals are:\n", normals_3a)
            print("Shape of ptsp1: ", pts_p1.shape)
            print("Shape of ptsp2: ", pts_p2.shape)
            print("Shape of ptsp3: ", pts_p3.shape)
            print("Shape of ptsp4: ", pts_p4.shape)
            print("Shape of ptsp5: ", pts_p5.shape)
        
        n1 = normals_3a[0,:]
        n2 = normals_3a[1,:]
        n3 = normals_3a[2,:]
        n4 = normals_3a[3,:]
        n5 = normals_3a[4,:]
        
        proj_ptsp1 = project_and_normalize_pts(pts_p1.T).T
        proj_ptsp2 = project_and_normalize_pts(pts_p2.T).T
        proj_ptsp3 = project_and_normalize_pts(pts_p3.T).T
        proj_ptsp4 = project_and_normalize_pts(pts_p4.T).T
        proj_ptsp5 = project_and_normalize_pts(pts_p5.T).T

        print("Evaluating direction vectors for all points")

        start_time = time.time()
        K_3ai = np.linalg.inv(K_3a)
        d1 = K_3ai@proj_ptsp1        
        d2 = K_3ai@proj_ptsp2        
        d3 = K_3ai@proj_ptsp3        
        d4 = K_3ai@proj_ptsp4        
        d5 = K_3ai@proj_ptsp5
        end_time = time.time()

        print("Time taken: ", end_time-start_time)