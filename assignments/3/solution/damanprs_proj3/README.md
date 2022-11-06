# HW1
The basic command for running the code is

    python3 main.py --type <question_number>

## q1

### (A1)
    python3 main.py --type 1A1

- This prints the fundamental matrices for both teddy and chair iin the terminal window
- Point annotation dialog box, for chair opens up
- Select points
- New dialog box, with the epipolar lines in the second image opens up
- This repeats for the teddy images too

### (A2)
    python3 main.py --type 1A2

- This prints the essential matrices for the teddy and chair images on the terminal window 

### (B)
    python3 main.py --type 1B

- This prints the fundamental matrix from 7 point method, for the toybus on the terminal window
- A dialog box appears, similar to Q1A1, for point selection on toybus
- Follow similar procedure as Q1A1
- Repeat step 2 and 3 for toytrain

## q2
```
    python3 main.py --type 2 --ransac_object <object> --ransac_algorithm <algo_choice>
```
- Here the choices for ransac_object = ['toybus' , 'toytrain'] and for ransac_algorithm = ['7' , '8']
- Sample:
```
python3 main.py --type 2 --ransac_object toybus --ransac_algorithm 8
```
- This leads to the ransac iterations
- A graph between ratio of inlers v/s number of iterations comes up
- Closing it leads to a dialog box for the object coming up, for point annotation
- Annotate the points similar to Q1A1
- F matrix gets printed on the terminal window

## q3
```
    python3 main.py --type 3
```
-  A visualization of the point cloud pops up
-  Time taken by the main triangulation loop shows up on terminal

## q4
```
    python3 main.py --type 4
```
-  A visualization of the bad point cloud pops up
-  Close it, iterations for the non-linear least squares start up, takes around 70-80 sec
-  The final F matrix and the optimized point cloud pops up after the iterations

## q5
```
    python3 main.py --type 5 --q5_object <object>
```
-  object choices = ['car1' , 'car2']
-  Feature matches dialog window shows up
-  Close this, ransac starts up
-  follow steps from q2



### For more information

    python3 main.py --help