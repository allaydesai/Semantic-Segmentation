# CarND-Semantic-Segmentation

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Self-Driving Car Engineer Nanodegree Program

---

[//]: # (Image References)

[video]: ./images/result.gif "Video showing segmentation in action"
[plan_proc]: ./imgs/uu_000051.png "Planning Process"
[result1]: ./runs/1547934773.2136164/uu_000022.png "Result"
[result2]: ./runs/1547934773.2136164/uu_000027.png "Result"
[result3]: ./runs/1547934773.2136164/uu_000031.png "Result"
[result4]: ./runs/1547934773.2136164/uu_000063.png "Result"

### Overview

Goal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN) to acheive scene understanding. Semantic segmentation is the task of assigning meaning to part of an object by classifying each pixel location in categories such as road, car, pedestrian, sign, or any number of other classes. This helps us derive valuable information about every pixel in the image in comparision to traditional bounding box based approach of slicing sections. Purpose of scene understanding is to help develop perception which enables self driving cars to make decisions. 

![alt text][video]

### Goals:
- Helper functions: load_vgg, layers, optimize and train_nn implemented correctly
- Ensure passing all unit tests
- Model decreases loss over time
- Parameter tuning: batch size and epochs
- Model labels most pixels of roads close to the best solution (80% road, 20% of non-road acceptable)

### Project Files

The repository consists of the following files:
- /data : Folder containing the data 
- /examples : Sample results
- /run : result images of most recent run
- README.md : writeup file (either markdown or pdf)
- helper.py : Helper function for downloading vgg weights and batching
- main.py : Main program containing the model architecture and training
- project_test.py : Unit test for functions within main.py

### Dataset

Project uses the Kitti Road dataset which can be download from [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Setup and Dependecies

##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.

##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

---

### Model Details

The network built for this project is a Fully Convolutional Network (FCN8) split into an encoder and decoder implemented using tensorflow. FCN's comprise of 3 techniques:

- Replace fully connected layers with one by one convolutional layer: Matrix multiplication with spatial information.
- Up sampling through the use of transpose convolutional layers: Upsampling the previous layer to a higher resolution or dimension.
- Skipped connections: Skip connections allow the network to use information from multiple resolutions.

In `main.py`, you'll notice that layers 3, 4 and 7 of VGG16 are utilized in creating skip layers for a fully convolutional network. The reasons for this are contained in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf).

In section 4.3, and further under header "Skip Architectures for Segmentation" and Figure 3, they note these provided for 8x, 16x and 32x upsampling, respectively. Using each of these in their FCN-8s was the most effective architecture they found. 

The model predicts on each pixel in an image if its road or not. 

##### Tuning Parameters:

| Parameter | Value |
|-----------|-------|
| Image size | 160x576 |
| No of classes | 2 |
| Batch size | 16 |
| Epochs | 110 |
| Learning rate | 0.0001 |
| Keep prob | 0.75 |

### Results

I first trained the model for 100 epochs and acheived a loss of 1.025. I observed that the loss was yet decreasing and I may have ended training too soon. The resulting images were good for most part but there were still cases where some cars may get classified as part of road or there are holes in detection of regions which should be road. Hence, I retrained the model this time for 110 Epochs and this time the loss was 0.XXX. Some of the resulting test images can be seen below:

![alt text][result1] | ![alt text][result1] 
:-------------------------:|:-------------------------:
![alt text][result1] | ![alt text][result1] 


### Conclusion

There are many areas for improvement in this project. Starting with the dataset. The dataset used for this project comprises of only 208 images out of which only 192 were used for training. This is a pretty small dataset hence performance of the model will be limited to similar scenes. Thus a larger dataset would help the model generalize better. Another possible improvement would be add augmentation to the training images. This would increase the size of the dataset and help model generalize. 

