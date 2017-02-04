Behaviour Cloning P3 project - Oren Meiri

Network Architecture
--------------------

I have based my solution on the Nvidia network.

Conv layer: 24 outputs. conv 5 x 5 (stride 2 x 2)
RELU activation (for non linearity)

Conv layer: 36 outputs. conv 5 x 5 (stride 2 x 2)
RELU activation

Conv layer: 48 outputs. conv 5 x 5 (stride 2 x 2)
RELU activation

Conv layer: 64 outputs. conv 3 x 3
RELU activation

Conv layer: 64 outputs. conv 3 x 3
RELU activation

Flatten
Dropout: 50%

FC 100
RELU activation
Dropout: 50%

FC 50
RELU activation

FC 10
RELU activation

FC 1 - output

Training was performed with Adam optimizer & MSE loss function (see discussion below)
I reduced number parameters in FC layers (compared to Nvidia architecture) because I have very limited number of samples otherwise the network will quickly overfit.

Dataset
-------

I have used Udacity set of 8K samples.  I have not recorded my own training data because using keyboard produces non-smooth steering angle values.  I didn’t manage to stay on track while controlling with mouse.  With augmentation techniques discussed below this training set was used to generate infinite amount of samples, both for training and validation.

I have resized the original images (160 x 320) to (80, 160) to ease network training time and get it closer to Nvidia input size (66 x 200). I believe the network architecture is optimized to this image input size.

See imageX_before.jpg and imageX_after.jpg for example augmented samples

Training Approach
-----------------

Keras Generator:
During training I need to repeatedly feed samples for the network to learn from.  Fitting all images of the training set into memory is not feasible so I used Keras fit_generator.  I provide the training algorithm a function that provides samples on the fly.  Each time it provides a batch of samples and the previous batch can be erased from memory until it is needed in the next epoch.  Using this approach, in each iteration, I augment the images slightly differently (based on randomly generated parameters), thus providing wide range of samples to train on.

Augmentation:
To generate large number of samples from the limited training set and improve model generalization for dealing with the more challenging track 2, I tried the following augmentation techniques.  Some didn’t provide value so didn’t remain in the model.

Convert to grayscale - change input RGB images to grayscale images.  DELETED

Convert to HSV - change input images from RGB to HSV.  DELETED

Use left / right images from training set - To help the network learn how to return from drifting to a side, I used the left & right images with corresponding adjustment to steering angle to return the car to the center of the track

Flip right/left - Training data has bias to left turns to I randomly flipped 50% of samples to introduce equal number of samples with right/left turns

Shear - Using shear method I augment the road to appear to turn more sharply to left/right (and adjust steering angle accordingly).  This helped the model deal with very sharp corners present in track 2 but were not available in training set to learn from.

Adjust brightness - Track 2 is much darker.  I randomly adjusted whole image brightness and this greatly helped dealing with track 2.

Dropout - To avoid overfitting, I used dropout.  This helps the network generalize better and not rely on specific features it sees in training set and rely too much on them.

MAE/MSE - the network needs to predict steering angle which is in range -0.5, 0.5.  I believe in this case, MSE panelize small prediction error more than large error.  I tried MAE but wanted to panelize large errors exponentially so I returned to MSE and multiplied steering angle by 100 to reach larger values.  I remembered to divide the output predictions by 100 when feeding simulator in autonomous mode. DELETED MAE

Save model after every epoch - I saved the model after each epoch so that I could choose which set of weights to run with the simulator.  This way I could test the network (on simulator) before all epochs finished.  I could also choose to run set of weights before the network started overfitting.



