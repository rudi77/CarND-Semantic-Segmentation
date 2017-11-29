# Semantic Segmentation

<a href="url"><img src="./images/cross_entropy.png" height="400" width="1024" ></a>

### Introduction
This is the second project of the last term of the self driving car nanodegree. The goal of this project is to train a [Fully Convolutional Network](https://arxiv.org/abs/1411.4038) (FCN) that is able to detect the pixels of a road in images. 
An FCN is a CNN without any fully-connected layers. In this project we used pretrained VGG16 model, replaced the fcn layers with 1x1 convolutional layers and finally added several deconvolution or upsampling layers. The result is fully convolutional network which is able to do a semantic pixel segmentation in provided images.

### Program Usage


### Setup

##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - 
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
