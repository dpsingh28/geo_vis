#!/usr/bin/env python3

from cv2 import destroyAllWindows
import annotate
from annotate import cv2
from annotate import np
from utils import MyWarp
import rectifications
import argparse

def get_projective_line(p1 , p2):
    p1 = np.append(np.flip(p1) , 1)
    p2 = np.append(np.flip(p2) , 1)
    line = np.expand_dims(np.cross(p1,p2) ,axis=0)
    print(line)
    return line

def new_color():
    color = np.random.randint(0 , 255 , size=3 , dtype=int)
    return color.tolist()


def make_lines(image_path, points):
    
    lines = np.reshape(np.array([] , dtype=int) , (0,3))
    img = cv2.imread(image_path)
    if(len(points)%2 !=0):
        print("Number of points selected not even")
        return
    for i in range(int(len(points) /2)):
        p1  = points[2*i,:]
        p2 = points[2*i +1,:]
        if (i%2 ==0):
            color = new_color()
        line_img = cv2.line(img , p1, p2 , color , 2)
        lines = np.append(lines , get_projective_line(p1,p2) ,axis =0)

    cv2.imshow("Line" , line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lines

def get_points(image_path):
    points = annotate.annotate(image_path)
    print("Points received: \n", points )
    return points

def process_image(args1):
    image_path1 = args1.image_path1
    if(image_path1 == None):
        raise Exception("provided image_path1 is null")
    
    points = get_points(image_path1)
    lines = np.asarray(make_lines(image_path1 , points) , dtype=float)
    print("Lines recieved: \n" , lines)
    lines = lines / np.reshape(lines[:,2] , (-1,1))
    return lines

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
        lines = process_image(args)
        Ha = rectifications.affine(lines)
        print(Ha)
        orig_img  = cv2.imread(args.image_path1)
        new_img = MyWarp(orig_img , Ha)
        cv2.imwrite('processed_images/affine/' + args.output_name + '.jpg' , new_img)
        cv2.imshow('Affine' , new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    elif(qtype == 'q2'):
        print("Metric rectification")
        lines = process_image(args)
        print(lines)
        
        Hp = rectifications.metric(lines)
        orig_img_m  = cv2.imread(args.image_path1)
        new_img_m = MyWarp(orig_img_m , Hp)
        cv2.imshow('Metric' , new_img_m)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        

    elif(qtype == 'q3'):
        print("Homography estimation")
    else:
        print("No such case")
    
    
    