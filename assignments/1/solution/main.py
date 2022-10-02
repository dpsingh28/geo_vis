#!/usr/bin/env python3

from cv2 import destroyAllWindows
import annotate
from annotate import cv2
from annotate import np
from utils import MyWarp
from utils import cosine
import rectifications
import argparse

def get_projective_line(p1 , p2):
    p1 = np.append(np.flip(p1) , 1)
    p2 = np.append(np.flip(p2) , 1)
    line = np.expand_dims(np.cross(p1,p2) ,axis=0)
    print("Line equation: ",line)
    return line

def flip_color(color):
    if color == (255,0,0):
        return (0,255,0)
    elif color == (0,255,0):
        return (255,0,0)

def make_lines(img, points, out_name):
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

        line_img = cv2.line(img , p1, p2 , color , 2)
        lines = np.append(lines , get_projective_line(p1,p2) ,axis =0)

    cv2.imshow("Line" , line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('processed_images/annotations/' + out_name + '.jpg' , line_img)

    return lines

def get_points(image):
    points = annotate.annotate(image)
    print("Points received: \n", points )
    return points

def process_image(img, out_name):
    
    points = get_points(img)
    lines = np.asarray(make_lines(img , points  , out_name) , dtype=float)
    print("Lines recieved: \n" , lines)
    lines = lines / np.reshape(lines[:,2] , (-1,1))
    return lines

def show_points(img,pts):
    pt_img = img.copy()
    for i in range(len(pts)):
        pt_img = cv2.circle(pt_img, tuple(pts[i,:]), radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow("Annotated Points" , pt_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('processed_images/annotations/q3_annotations.jpg' , pt_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main file for assignment 1 of 16822 Geometry-Based Vision")
    parser.add_argument('--type' , type=str , required=True , choices=['q1' , 'q2' , 'q3'])
    parser.add_argument('--image_path1' , type=str , required=False)
    parser.add_argument('--image_path2' , type=str , required=False)
    parser.add_argument('--output_name' , type=str , required=False)
    
    args = parser.parse_args()
    qtype = args.type

    if(qtype == 'q1'):
        print("Affine rectification")
        img1 = cv2.imread(args.image_path1)
        lines = process_image(img1 , args.output_name)
        print("Cosine between first set original: ", cosine(lines[0,:] , lines[1,:]))
        print("Cosine between second set original: ", cosine(lines[2,:] , lines[3,:]))
        Ha = rectifications.affine(lines)

        orig_img  = cv2.imread(args.image_path1)
        new_img = MyWarp(orig_img , Ha)
        
        cv2.imshow('Affine' , new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('processed_images/affine/' + args.output_name + '.jpg' , new_img)

        lines_rect = process_image(new_img , args.output_name + '_affine_rect_lines')
        print("Cosine between first set rectified: ", cosine(lines_rect[0,:] , lines_rect[1,:]))
        print("Cosine between second set rectified: ", cosine(lines_rect[2,:] , lines_rect[3,:]))



    elif(qtype == 'q2'):
        print("Metric rectification")
        img1 = cv2.imread(args.image_path1)
        lines = process_image(img1 , args.output_name)
        print("Cosine between first set original: ", cosine(lines[0,:] , lines[1,:]))
        print("Cosine between second set original: ", cosine(lines[2,:] , lines[3,:]))

        Hp = rectifications.metric(lines)
        orig_img_m  = cv2.imread(args.image_path1)
        new_img_m = MyWarp(orig_img_m , Hp)

        cv2.imshow('Metric' , new_img_m)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('processed_images/metric/' + args.output_name + '.jpg' , new_img_m)


        lines_rect = process_image(new_img_m , args.output_name + '_metric_rect_lines')
        print("Cosine between first set rectified: ", cosine(lines_rect[0,:] , lines_rect[1,:]))
        print("Cosine between second set rectified: ", cosine(lines_rect[2,:] , lines_rect[3,:]))
        
        

    elif(qtype == 'q3'):
        print("Homography estimation")
        img_path1 = args.image_path1
        img_path2 = args.image_path2
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        pts1 = get_points(img1)
        pts2 = get_points(img2)
        show_points(img2 , pts2)

        Hh , pts1_homo = rectifications.homography(pts1 , pts2)
        new_img1 = MyWarp(img1 , Hh)
        warped_img1 = np.zeros_like(img2)

        #Check first point in warped image
        nonzero_pts = np.transpose(np.nonzero(new_img1))
        nonzero_pts = nonzero_pts[:, 0:2]
        first_pixel = nonzero_pts[0,:]
        ##

        pts1np = np.fliplr(pts1)
        pts2np = np.fliplr(pts2)

        warped_img1[pts2np[0,0]- first_pixel[0]:pts2np[0,0]+new_img1.shape[0] - first_pixel[0], pts2np[0,1]- first_pixel[1]: pts2np[0,1]+new_img1.shape[1] - first_pixel[1] , :] = new_img1
        warped_nonzero = np.nonzero(warped_img1)
        c1_nonzero = (warped_nonzero[0] , warped_nonzero[1] , np.zeros_like(warped_nonzero[1]))
        c2_nonzero = (warped_nonzero[0] , warped_nonzero[1] , np.ones_like(warped_nonzero[1]))
        c3_nonzero = (warped_nonzero[0] , warped_nonzero[1] , 2*np.ones_like(warped_nonzero[1]))
        
        ##removing pixels in image 2
        img2[c1_nonzero] = 0
        img2[c2_nonzero] = 0
        img2[c3_nonzero] = 0

        combined_img = cv2.add(img2 ,warped_img1 )
        cv2.imshow('combined' , combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('processed_images/annotations/q3_final.jpg' , combined_img)


    else:
        print("No such case")
    
    
    