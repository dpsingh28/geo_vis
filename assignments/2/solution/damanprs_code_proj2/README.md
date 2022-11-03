# HW1
The basic command for running the code is

    python3 main.py --type <question_number>

## q1

### (a)
    python3 main.py --type 1a

- This opens up a visualization of the bunny with the 3D points overlayed on it
- Pressing any key gives another visualization of the bounding box on the bunny
- The camera matrix gets printed on the terminal screen
- The visualizations get saved in the submissions/ directory

### (b)
    python3 main.py --type 1b

- This opens up a dialog box, with the custom cuboid image
- Annotate 6 points on this cuboid, to make the co-ordinate system
- Annotations must be in accordance with what is show in the handout
  - 0 on bottom-center
  - 1 on bottom-right
  - 2 on bottom-left
  - 3 on top-center
  - 4 on top-right
  - 5 on top-left
- Press any key after annotations are done
- The camera matrix gets printed on the terminal screen
- Another visualization pops up, with the front two faces having their boundary outlined
- This final visualization gets saved in the submissions/ directory
  
## q2
### (a)
```
    python3 main.py --type 2a
```
  - This opens up a dialog box, with the pre-provided annotations
  - Press any key. A visualization of vanishing points appears
  - Press any key again. Visualization of vanishing points with the connecting lines appears
  - Press any key, visualization goes away, K matrix gets printed on terminal screen
  - The final visualization gets saved in the submissions/ directory
  

  ### (b)
```
    python3 main.py --type 2b
```
  - Dialog box, with pre-provided annotations pops up
  - Press any key. Set of parallel lines appear
  - Press any key. K matrix and angles gets printed on terminal screen
  
  ### (c)
```
    python3 main.py --type 2c
```
  - Custom image appears. Annotate points, starting from top left and going clockwise, for one rectangl at a time
  - Do this for all 3 rectangles
  - Dialog box, with pre-provided annotations pops up
  - Press any key. Set of parallel lines appear
  - Press any key. K matrix and angles gets printed on terminal screen

## q3
### (a)
```
    python3 main.py --type 3a --first_run {True,False}
```
-  first_run variable is required to run q3                                               
	- give False if intrinsics, normals and plane points npy files exist in submissions directory                                                                                                
	- give True if this is the first run. All values will be saved in submissions directory
- First visualization is the one pre-provided
- Press key top get out of the first viz. If first run, a dialog box will come up annotate points for parallel lines, to get vanishing points
- Press key, get lines visualized
- Press any key, K matrix and other relevant information gets printed on terminal screen
- matplotlib scatter plot shows up


### For more information

    python3 main.py --help