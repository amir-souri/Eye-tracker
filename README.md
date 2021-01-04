**Making an eye tracker**

The purpose of this project is to gain practical experience with the use of computer vision and simple machine learning
techniques. The goal is to create an eye tracker. 

1. Pupil and glint detection: I will implement methods to detect pupils and glints in images of eyes.
2. Simple gaze estimation: I will use the pupil detector to implement a gaze estimation system.


An eye tracker refers to a system including camera, IR light sources and software that detects where the user is looking (gaze).
Without getting into too much detail, the gaze is directly related to the relative position of the screen, eye orientation, 
and camera orientation. When looking at an object, the person orients the eye so that the light of the object falls onto a 
small region on the retina, called the Fovea. The position of the fovea varies between people and has to be determined by calibration. 
I will use the pupil to determine the rotation of the eye and implicitly learn the direction of attention (called the visual axis) by calibration.
All the data needed for calibration and testing is provided in the inputs/images directory.

Setup:
The head of the subject rests on a stand and the camera has a relatively fixed position on the
table. During recording, the subject is asked to look at a number of red dots on the screen moving in a specific pattern.


Directory structure:
The recorded eye images and corresponding targets on the screen are saved in a number of subfolders in the inputs/images directory. 
Each subdirectory contain an image sequence of which the first 9 are calibration images. Each subdirectory additionally contains 
these three data files: positions.json , pupils.json , and glints.json.

•positions.json contains gaze coordinates (in pixel coordinates) for each image.
•pupils.json contains annotated ellipse parameters for the pupil in each image.
•glints.json contains annotated glint parameters for each image.

The aim is to accurately infer the gaze (as position on the screen) of the user, e.g. the position on the screen given the eye image. 
This is achieved by detecting the pupil position in the image and the using a number of sample images with known gaze
coordinates to create a model f_θ ( x, y ) (using regression) that maps image pupil positions x, y to screen positions x_0 , y_0 . 
Additionally, I will use glints, which are reflections caused by infrared LEDs placed at the screen’s corners, to normalise 
the eye position in the image and hence correct for errors related to movements of the subject.



- Pupil and glint detection
Create robust detectors for pupils (1) and glints(2). 
The consistency and accuracy of the gaze estimation part depends directly on the quality of the detection algorithms.
The implementation for this task is provided in detector.py . 
The script detector_viz.py is used to visualise the results.
The test_detector.py script takes one command-line argument which specifies which subfolder in the inputs/images/ directory to use. 
One can run it as python detector_test.py <folder> . The script displays a window with a slider which selects the image to use for testing. 
All samples are annotated with the correct pupil and glint positions.

Task 1: Pupil detection
All the code for this task is written in the find_pupil function in detector.py .
The pupil is roughly a circular hole that lets light into the eye.
Most light is absorbed by the photosensitive cells (retina) on the back of the eye, giving the pupil its black appearance. 
The pupil appears approximately elliptical in images because of the projection onto the image. I model the
pupil shape as an ellipse with the following parameters ( x, y, a, b, θ ) and assume the intensity to be darker than the surroundings.



Approach:
The general approach is to:
1. Threshold the image to find BLOBs.
2. Find contours from BLOBs.
3. Select a single contour candidate as the best.
4. Fit ellipse to the selected contour candidate.

Detecting the pupil is fairly straight forward using thresholding since it is much darker than its surroundings.
Thresholding should produce BLOB candidates which can then be fitted to ellipses.


Task 2: Glint detection
Design a method for detecting glints in eye images.


Task 3: Evaluation
Test the actual performance of the detectors. The code for this task is provided in test_detector.py .

Calculate distances for pupils: For each image, calculate the distance (using the provided function dist from utils.py ) between the 
ground-truth pupil centre and the one detected using your implementation.

Calculate distances for glints: For each detected glint, measure the distance to the closest ground-truth glint and
record this as the error. Because I do not differentiate between the different glints, I choose the closest as the most
likely candidate. 

Data analysis: For both pupil and glints calculate at least the mean and median distance. Also produce a histogram plot for
each.

- Gaze estimation

The goal is to estimate the gaze of image sequences using a regression model. As mentioned in the introduction, each
image sequence contains 9 images for calibration and a varying number of images for inference. The calibration samples always
represent the same 9 screen positions which form a simple 3 by 3 grid.
For each sequence, I will use the 9 calibration samples to train a regression model and then use the model to predict gaze positions
for the rest of the images.

positions.json contains the ground-truth gaze positions for each image as an array (stored as y, x for each point). 
The included image sequences (found in inputs/images ) are divided into two groups:


• No head movement:
pattern0
• Head movement and rotation:
pattern1
pattern2
movement_medium
pattern3
movement_hard


The script, position_vis.py is used to visualise the image and corresponding screen point. It has one commandline
argument which is the subfolder (in inputs/images/ ) to use, e.g. python position_vis.py pattern1 .

Task 1: Basic gaze estimator
The mapping function f_θ ( x, y ) can take any form. Because the eye is spherical, the relationship between pupil
position in the image and gaze is non-linear. In this task, however, I will approximate the gaze mapping by a second-order polynomial function.
I assume the gaze coordinate x_0 , y_0 to be independent variables.
Therefore, I train a separate model for each. This is equivalent to training one model with a vector output.

Calibration: Learn the parameters φ for the polynomial regression using the calibration images and points aved in self.positions .
Use the detector to detect pupil points.
Use the centres of the detected pupils to create a design matrix. Then create two models, one for the X coordinates of the 
calibration points and one for the Y coordinates.

Estimation: Implement the estimate function which predicts the gaze given an eye image using the learned model. 
First, detect the pupil centre in the input image. Then calculate and return the estimated screen coordinates using the models created
during calibration.

Task 2: Evaluation
The code for this task is provided in test_gaze.py .

Calculate distance: For each dataset, create and calibrate
a GazeModel and use it to estimate the gaze position for the
points not used for calibration. Calculate the euclidean distance
between the estimate and ground-truth gaze position for each image and and save it.

Data analysis: Perform the analysis for the sequences with
and without head movement separately. Calculate the mean and
median distance. Also produce a histogram, but this time use
the cumulative=True and density=True arguments to make a
cumulative histogram that is normalised, i.e. shows fractions
instead of number of occurrences.

Correlation: Record both the pupil detection distance and
gaze distance errors. Measure their correlation.

