# BlackboardNumberRecognition

BlackBoardNumberRecognition is a python application that automatically predicts handwritten digits. 
Users can manually draw single digits on a canvas and the application automatically predicts what number was drawn. 
The underlying model uses a machine learning algorithm trained on a dataset containg handwritten single digits.

## Table of Contents

[1. Introduction](#introduction)  
[2. Getting Started](#gettingStarted)  
[3. Current State](#todo)

<a name="introduction"/></a>
## Introduction

![Screenshot of the application](https://github.com/JannisGz/BlackboardNumberRecognition/blob/main/resources/screenshot.png)

The application includes a GUI, which acts as an input generator for the underlying model. It contains a 
blank black canvas and two buttons. The buttons are used to submit a drawing or to clear the canvas. Clicking and 
dragging the mouse cursor on the canvas leaves a white path. This can be used to draw a single digit number (0 - 9).

When an image is submitted, the model guesses which digit was drawn by the User. The prediction is displayed next to the canvas
along with a percentage of how certain the model is of its prediction.


<a name="gettingStarted"/></a>
## Getting started

The subdirectory <em>resources</em> contains a demo video. The application itself can be started by running the gui class with a 
python interpreter. During development version 3.8 of Python was used and tested with Mac OS. The external dependencies of this project are 
listed in the <em>requirements.txt</em> file.

When the application is started for the first time and no classification model is found, the application creates a new model. 
To create this model the MNIST data set is used. It contains a large number of images containing a single handwritten digit in white color 
on a black background. The associations between images and depicted numbers is used to train a CNN model.



<a name="todo"/></a>
## Current state of the project

Although the used classification model reaches accuracy rates of about 99% on test data, the same level of quality is not reached when using the GUI. 
A reason could be that the digit drawn on the canvas is unsimilar to the dataset used during training. For example, all images 
in the training data sets are almost the same size (filling most of the canvas) and placed in its center. 
Overall the application is able to predict images that are drawn in a similar fashion correctly in about 9 of 10 cases. 
The digits 9 and 6 seem to be hardest to predict correctly.
