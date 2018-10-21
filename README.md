       
# End-to-End Deep Learning for Self-Driving Cars

# Behavior Cloning Project

### Project Summary: 
Employ deep learning to clone user car driving behavior on Udacity's simulator. It is a supervised regression problem between the car steering angles and the road images in front of a car. The trainig images are taken from three different camera angles (from the center, the left and the right of the car). As the task involves image processing, the model employed is a Convolutional Neural Network (CNN) for automated feature engineering and developing invariance to translation/rotation. The architeture of the network employed is based off of the NVIDIA model, which has been proven to work in this problem domain but modifed using cues from VGG-net. 

The goals of this project are the following:
1. Use the provided Udacity simulator to collect data of good driving behavior
2. Build, a convolution neural network (CNN) in Keras that predicts steering angles from images
3. Train and validate the model with a training and validation set
4. Test that the model successfully drives around track one without leaving the road


## Details about included files

The project includes the following files:
* model_training_final.py containing the script to load and train the model
* augment_helper_fns.py containing the various functions used for implementing data augmentation.
* model_final.py containing the function to create a model and detailing the CNN architecture.
* drive.py for driving the car in autonomous mode (Added a couple of lines of code to import preprocessing function from the model_final script)
* model.h5 containing a trained CNN's weights


### `drive.py`

Using my trained model weights stored an h5 file, i.e. `model.h5`, the car can be driven autonomously around the track in Udacity's simulator using the `drive.py` by executing 
```sh
python drive.py model.h5 run_track
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection. The last argument, `run_track`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run_track --fps 48
```

Creates a video based on images found in the `run_track` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run_track.mp4`.

Optionally, the FPS (frames per second) of the video can be specified as an argument. For the above command, the video will run at 48 FPS. The default FPS is 60.


## Data collection

The data collection process involves steering a car around a track in Udacity's simulator (https://github.com/udacity/self-driving-car-sim) which possesses functionality to record the user driving behaviour by capturing multiple frames per second of driving a car in the simulator. The model utilizes image data and steering angles to train the CNN and is later deployed to drive the car autonomously around the track by providing steering angle as output to an autonomous vehicle.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving by driving around the first track twice and then complemented this with more driving around difficult sections like sharp curves, bridge, sections of road without border. I also included data exhibiting recovery scenarios from the left and right sides of the road. 

**For more details about how I created the training data, see the last section.**

## Model Training Strategy

The acquired knowledge about Convolutional Neural Networks (CNNs) is exploited to clone (user) driving behavior. I trained, validated and tested a model in Keras using NVIDIA's model (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) as a starting point. 

The network model `model_final.py` is that of a feed-forward CNN. The convolutional layers were modified taking cue from VGG-Net. It consists of 3x3 convolution filters and depths between 24 and 128. The model includes ELU activation layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Maxpooling with a stride of 2 is utilized rather than convolution using a stride of 2 since it resulted in better learning/training.

The `model_training_final.py` file contains source code for training and saving the Convolution Neural Network (CNN). It provides the utility of using a generator, if needed, for training the network in the event the training data is infeasible to store in memory for training.

*P.S- I have not used a generator for training the model since the training data was not enormous or memory-intensive and enabled 7-10x faster training than using a generator. But it can be utilized by simply setting the 'use_generator' variable in the `model_training_final.py` file to true*

The `model_final.py` file describes the CNN architecture and generates the model. 

#### Network training parameters

To train the final model, I used the following hyperparameter configuration-

Optimizer: Adam optimizer with initial learning rate: 1e-4 <br />
Batch size: 64 <br />
Epochs: 10 <br />

Training for epochs is capped at 10 epochs. Training the model for more than 10 epochs led to lower error on the training and validation set but made the car steer more sensitive to small variations and led to car going astray. Whereas, training for epochs less than 10 epochs did not enable the network to learn (fit) the data properly as evidenced by comparitively higher error and car driving off the road at the corners.


## Model Architecture

The overall strategy for deriving a model architecture was to design a model powerful enough to result in a sufficiently low validation error (< 5%) through training over a few epochs (about 10) without overfitting.

My first step was to use a CNN model similar to the one which was deployed by NVIDIA as described in their paper titled 'End to End Learning for Self-Driving Cars'. I thought this model might be appropriate because it had proved to possess capability for modelling the data. Moreover, I wanted to benchmark the performance of this network so that it would serve as a guide to enable improvements to the CNN architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

#### Attempts to reduce overfitting in the model

The model contains a single dropout layer after the convolutional layers to reduce overfitting. I experimented with using dropout layers interspersed between fully connected layers but it resulted in much higher loss & inhibited the capability of the network to learn. I also experimented with implementing Batch Normalization between convolutional layers. The loss ended up being almost similar to single dropout layer but the car performance on the challenge track was much worse.

#### Hyperparameter tuning
i.   The lower learning rate was obtained after hyperparameter tuning. It ensured that the network was able to learn at a moderate pace without much oscillation (characteristic of a high learning rate for any application). <br />
ii.  The dropout drop probability of 0.50 after the Convolution stages was arrived at empirically through experimentation. <br />
iii. I also experimented with filter size & padding type for convolutional layers, number of convolutional & fully connected layers, and number of hidden units in the fully connected layers to end up with the final architecture for my CNN.

The model was trained and validated on different data sets to ensure that the model was not overfitting. Moreover, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

At the end of the process, the vehicle is able to drive autonomously around track one very smoothly, *almost as good as me!* <br />
**What's more amazing is that the model is robust enough to enable the car to drive completely autonomously around the challenge track too** * <br />
*_Albeit hitting the street lamp like structures in 2 parts of the track, requiring manual intervention. Nothing that a little more data for these troublesome areas wouldn't fix_

### Final Model Architecture

The final model architecture `model_final.py` consisted of a CNN with the following layers and layer sizes:

#### Pipeline for the model

Input	64x64x3 - Preprocessed (Cropped, resized, color space transformed: RGB -> YUV) image

#### Pipeline for Convolutional layers

The number of convolution filters increase from 24 to 48 over 3 stages <br />
**Stage 1 to 3:** <br />
5x5 Convolution  +  ELU Activation <br />
2x2 MaxPool <br />
**Output: 8 x 8 x 48**

**Stage 4:** <br />
3x3 Convolution ('VALID' padding) +  ELU Activation <br />
**Output: 6 x 6 x 64**

**Stage 5:** <br />
3x3 Convolution ('VALID' padding)  +  ELU Activation <br />
Dropout(0.50)        Probability of dropping a neuron -> 0.50 <br />
Flatten              
**Output: 4 * 4 * 64**

##### Pipeline for Fully Connected layers
**Stage 6:** <br />
Fully Connected + ELU Activation (Input: 4 * 4 * 64, Output: 100)

**Stage 7:** <br />
Fully Connected + ELU Activation (Input: 100, Output: 50)

**Stage 8:** <br />
Fully Connected + ELU Activation (Input: 50, Output: 10)

**Stage 9:** <br />
Fully Connected    (Input: 10, Output: 1)    --> steering angle (Network Output)


## Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_center_lane.jpg?raw=true "Center lane driving")

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to be able to stay on the road even if it wanders off near the edges of the road. These images show what a typical recovery looks like :

![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_recovery1.jpg?raw=true "Recovery start")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_recovery2.jpg?raw=true "Mid-recovery")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_recovery3.jpg?raw=true "Recovered")

To augment the dataset, I flipped images and angles thinking that this would help the model generalize to both types of curves since track one had predominantly left curves. For example, here is an image that has then been flipped:

![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_aug_og.jpg?raw=true "Image prior to augmentation")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_aug_flip.jpg?raw=true "Flipped image")

I also augmented the dataset with minor translations in the x and y directions, brightness scaling to simulate different lighting conditions & randomly scaling brightness of only a certain portion of the image to emulate shadows on the road.
These transformations can be observed in the images below:

![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_aug_transl.jpg?raw=true "Translated image")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_aug_bright.jpg?raw=true "Brightness augmented image")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_aug_shadow.jpg?raw=true "Shadow augmented image")

Then I recorded center lane driving behaviour on sections of track two which were unique to it in order to get more data points and help the network generalize better. Here are a few examples of these peculiar & challenging areas of track two:

![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_jungle1.jpg?raw=true "Jungle image")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_jungle2.jpg?raw=true "Jungle image 2")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_jungle3.jpg?raw=true "Jungle image 3")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_jungle4.jpg?raw=true "Jungle image 4")
![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_jungle5.jpg?raw=true "Jungle image 5")

After the collection process, I had 41,733 number of data points (including left & right camera views). 
But for each frame, I randomly chose between the left, right & center camera views which meant I effectively had 13,911 
data points. I then preprocessed this data by:
1. Cropping the top & bottom of the image to avoid confusion resulting from the surroundings,
2. Resizing image to 64 x 64 to reduce number of parameters required without affecting network perfromance, and 
3. Transforming the color space from RGB to YUV (as was done in the NVIDIA paper).

The following shows the 3 channels of image post the processing pipeline:

![alt text](https://github.com/sbanga16k/End-to-End-Learning-for-Self-driving-cars/blob/master/Result_images/img_preprocessed.JPG?raw=true "Preprocessed image")

I finally randomly shuffled the data set and set apart 20% of the data for the validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by extensive hyperparameter tuning. I used an Adam optimizer so that manually changing the learning rate wasn't necessary.
