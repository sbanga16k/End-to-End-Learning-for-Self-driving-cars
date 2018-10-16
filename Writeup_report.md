
# **Behavioral Cloning** 

## Writeup

### Behavioral Cloning Project

The goals of this project are the following:
1. Use the simulator to collect data of good driving behavior
2. Build, a convolution neural network (CNN) in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road
5. Summarize the results with a written report


[//]: # (Image References)

[image1]: ./disp_images/img_center_lane.jpg "Center lane driving"
[image2]: ./disp_images/img_recovery1.jpg "Recovery start"
[image3]: ./disp_images/img_recovery2.jpg "Mid-recovery"
[image4]: ./disp_images/img_recovery3.jpg "Recovered"
[image5]: ./disp_images/img_aug_og.jpg "Image prior to augmentation"
[image6]: ./disp_images/img_aug_flip.jpg "Flipped image"
[image7]: ./disp_images/img_aug_transl.jpg "Translated image"
[image8]: ./disp_images/img_aug_bright.jpg "Brightness augmented image"
[image9]: ./disp_images/img_aug_shadow.jpg "Shadow augmented image"
[image10]: ./disp_images/img_jungle1.jpg "Jungle image"
[image11]: ./disp_images/img_jungle2.jpg "Jungle image 2"
[image12]: ./disp_images/img_jungle3.jpg "Jungle image 3"
[image13]: ./disp_images/img_jungle4.jpg "Jungle image 4"
[image14]: ./disp_images/img_jungle5.jpg "Jungle image 5"
[image15]: ./disp_images/img_preprocessed.jpg "Preprocessed image"

## Rubric Points

#### ** Here I will consider the rubric points individually and describe how I addressed each point in my implementation **

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_training_final.py containing the script to load and train the model
* augment_helper_fns.py containing the various functions used for implementing data augmentation.
* model_final.py containing the function to create a model and detailing the CNN architecture.
* drive.py for driving the car in autonomous mode (Added a couple of lines of code to import preprocessing function from the model_final script)
* model.h5 containing a trained CNN's weights 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model_training_final.py file contains highly modular and organized code for training and saving the convolution neural network. It shows the pipeline I used for training and validating the model, and it contains accompanying comments to explain how the code works. It provides the utility of using a generator, if needed, for training the network in the event the training data is infeasible to store in memory for training.

** P.S- I have not used a generator for training the model since the training data was not enormous or memory-intensive and enabled 7-10x faster training than using a generator. But it can be utilized by simply setting the 'use_generator' variable in the model_training_final.py file to true.**

The model_final.py file describing the network architecture and generating the model. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The network model is that of a feed-forward CNN. The base architecture was taken as the NVIDIA model & the convolutional layers were modified taking cue from VGG-Net (model_final.py)
It consists of 3x3 convolution filters and depths between 24 and 128. The model includes ELU activation layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Maxpooling with a stride of 2 is utilized rather than convolution using a stride of 2 since it resulted in better learning/training.

#### 2. Attempts to reduce overfitting in the model

The model contains a single dropout layer after the convolutional layers to reduce overfitting. I experimented with using dropout layers interspersed between fully connected layers but it resulted in much higher loss & inhibited the capability of the network to learn.
I also experimented with implementing Batch Normalization between convolutional layers. The loss ended up being almost similar to single dropout layer but the car performance on the challenge track was much worse.

The model was trained and validated on different data sets to ensure that the model was not overfitting. Moreover, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

To train the final model, I used the following hyperparameter configuration-

Optimizer: Adam optimizer with initial learning rate: 0.0001 (1e-4)
Batch size: 64
Epochs: 10
Dropout- Prob_drop: 0.5 (After Stage 5)

i.   Adam optimizer was chosen since it automatically handles the learning rate decay, without explicit parameters and has proven to be one of the best optimizers for most applications in recent times in terms of achieving convergence. 
ii.  The lower learning rate was obtained after hyperparameter tuning. It ensured that the network was able to learn at a moderate pace without much oscillation (characteristic of a high learning rate for any application). 
iii. A batch size of 64 was ideal for the augmented training dataset which had a large number of variations.
iv. The number of epochs was finalized through hyperparameter tuning. Training for epochs higher than 10 led to lower error on the training & validation set but made the car steer more sensitive to small variations and led to car going astray. Whereas, training for epochs less 10 than did not enable the network to learn (fit) the data properly as evidenced by comparitively higher error and car driving off the road at the corners.
v. The dropout drop probability of 0.50 after the Convolution stages was arrived at after a lot of hyperparameter tuning.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving by driving around the first track twice and then complemented this with more driving around difficult sections like sharp curves, bridge, road without border. I also included data demonstrating recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design a model powerful enough to result in a sufficiently low validation error (< 5%) through training over a few epochs (about 10) without overfitting.

My first step was to use a CNN model similar to the one which was deployed by NVIDIA as described in their paper titled 'End to End Learning for Self-Driving Cars'. I thought this model might be appropriate because it had proved to possess capability for modelling the data. Moreover, I wanted to benchmark the performance of this network so that it would serve as a guide to enable improvements to the CNN architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include a dropout layer with a drop probability of 0.25 after the last convolutional layer. This resulted in much higher training & validation error, accompanied by worse performance on the simulator track. I arrived at a drop probability of 0.50 after considerable hyperparameter tuning. 
I experimented with dropout layers between fully connected layers of the CNN but it had an adverse effect on the network performance. I also tried Batch Normlization between convolutional layers but it wasn't able to combat the overfitting that much. So, I kept the first configuration of using a single dropout layer after the convolutional layers.

I also experimented with filter size & padding type for convolutional layers, number of convolutional & fully connected layers, and number of hidden units in the fully connected layers to end up with the final architecture for my CNN.

At the end of the process, the vehicle is able to drive autonomously around track one very smoothly, almost as good as me!
What's more amazing is that the model is robust enough to enable the car to drive completely autonomously around the challenge track too (albeit hitting the street lamp like structures in 2 parts of the track, requiring manual intervention). Nothing that a little more data for these troublesome areas wouldn't fix.

#### 2. Final Model Architecture

The final model architecture (model_final.py) consisted of a CNN with the following layers and layer sizes:

#### Pipeline for the model

Input	             64x64x3 - Preprocessed (Cropped, resized, color space transformed: RGB -> YUV) image

##### Pipeline for Convolutional layers

Stage 1:
5x5 Convolution      Output: 64 x 64 x 24 (1x1 stride, 'SAME' padding)  +  ELU Activation
2x2 MaxPool          Output: 32 x 32 x 24 (2x2 stride, 'VALID' padding)

Stage 2:
5x5 Convolution      Output: 32 x 32 x 36 (1x1 stride, 'SAME' padding)  +  ELU Activation
2x2 MaxPool          Output: 16 x 16 x 36 (2x2 stride, 'VALID' padding)

Stage 3:
5x5 Convolution      Output: 16 x 16 x 48 (1x1 stride, 'SAME' padding)  +  ELU Activation
2x2 MaxPool          Output: 8 x 8 x 48 (2x2 stride, 'VALID' padding)

Stage 4:
3x3 Convolution      Output: 6 x 6 x 64 (1x1 stride, 'VALID' padding)  +  ELU Activation

Stage 5:
3x3 Convolution      Output: 4 x 4 x 64 (1x1 stride, 'VALID' padding)  +  ELU Activation
Dropout(0.50)        Probability of dropping a neuron -> 0.50
Flatten              Input: 4 x 4 x 64, Output: 4 * 4 * 64

##### Pipeline for Fully Connected layers
Stage 6:
Fully Connected + ELU Activation (Input: 4 * 4 * 64, Output: 100)

Stage 7:
Fully Connected + ELU Activation (Input: 100, Output: 50)

Stage 8:
Fully Connected + ELU Activation (Input: 50, Output: 10)

Stage 9:
Fully Connected                  (Input: 10, Output: 1)                --> steering angle (Network Output)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to be able to stay on the road even if it wanders off near the edges of the road. These images show what a typical recovery looks like :

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the dataset, I flipped images and angles thinking that this would help the model generalize to both types of curves since track one had predominantly left curves. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

I also augmented the dataset with minor translations in the x and y directions, brightness scaling to simulate different lighting conditions & randomly scaling brightness of only a certain portion of the image to emulate shadows on the road.
These transformations can be observed in the images below:

![alt text][image7]
![alt text][image8]
![alt text][image9]

Then I recorded center lane driving behaviour on sections of track two which were unique to it in order to get more data points and help the network generalize better. Here are a few examples of these peculiar & challenging areas of track two:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

After the collection process, I had 41,733 number of data points (including left & right camera views). 
But for each frame, I randomly chose between the left, right & center camera views which meant I effectively had 13,911 
data points. I then preprocessed this data by:
1. Cropping the top & bottom of the image to avoid confusion resulting from the surroundings,
2. Resizing image to 64 x 64 to reduce number of parameters required without affecting network perfromance, and 
3. Transforming the color space from RGB to YUV (as was done in the NVIDIA paper).

The following shows the 3 channels of image post the processing pipeline:

![alt text][image15]

I finally randomly shuffled the data set and set apart 20% of the data for the validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by extensive hyperparameter tuning. I used an Adam optimizer so that manually changing the learning rate wasn't necessary.
