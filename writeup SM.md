# **Traffic Sign Recognition** 

## Writeup by Simón Mijares


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cell3.png "Visualization"
[image2]: ./examples/finals/Psign1.jpg "Traffic Sign 1"
[image3]: ./examples/finals/Psign2.jpg "Traffic Sign 2"
[image4]: ./examples/finals/Psign3.jpg "Traffic Sign 3"
[image5]: ./examples/finals/Psign4.jpg "Traffic Sign 4"
[image6]: ./examples/finals/Psign5.jpg "Traffic Sign 5"
[image7]: ./examples/finals/Psign6.jpg "Traffic Sign 6"
[image8]: ./examples/DataDistribution.png "Data Histogram"
[image9]: ./examples/before6.png "Data Histogram"
[image10]: ./examples/after6.png "Data Histogram"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/simonmijares/CarNDT1P2)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **42**

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.

First of all, lets see how our data is distributes

![alt text][image8]

Here is an exploratory visualization of the data set. It shows the code used and the text asociated to a radom image taken from the data set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth and sixth code cell of the IPython notebook.

As a first step, I decided to keep the image colors but apply normalization. This proves to help the network to converge much quickly just setting the valid range between 0.1 and 0.9.

	Top = 255/.8
	temp = channel/Top
	temp+=.1
The Sixth code cell just allow us to see that only a range change and a minor compresion have been made , but the image almost will look identical.

![alt text][image9]
![alt text][image10]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.

To validate my model, I simply shuffle the training data. I did this by shuffle function used in previous assingments.

Since I reached the 96.19% accuracy there were no need to generate additional data.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eighth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:-----------------------:|:--------------------------------------------------------------:| 
| **Layer I** |
| 1.Input         		| 32x32x3 RGB image   						| 
| 2.Convolution 5x5 | 1x1 stride, same padding, outputs 28x28x6 	|
| 3.Sigmoid		|										|
| **Layer II** |
| Convolution 11x11| 1x1 stride, same padding, outputs 18x18x16  	|
| Sigmoid		| Activation								|
| Max Pooling		| 2x2 stride,  outputs 9x9x6					|
| **Layer III** |
| Flatten			| Output: 9x9x16 = 1296					|
| Fully Connected	| Output: 120								|
| Tanh			| Activation								|
| **Layer IV** |
| Fully Connected	| Output: 84								|
| Sigmoid		| Activation								|
| Fully Connected	| Output: 43								|


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used a learning rate of 0.001 and train with the Adam Optimizer, initially I tried to modify the learning rate but later found that the Optimizer work this by itself so this modification was discarted.

The batch size was set to 512 since the machine seems to work fine with this kind of load.

The epochs were changing over the proccess of modifying the network. Initially were about 1000 so I could watch the behavior in long term. Later the net start to converge much quickly, so it was modified to 200, but then the net actually start converging quite sooner and the EPOCH parameter was set finally to 75.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the fourteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 96.2
* validation set accuracy of 94.6
* test set accuracy of 100

A well known architecture was chosen:

* What architecture was chosen?
   The architecture used is basically the propoused Lenet, changes were made only on the activation functions and one of the Max Pool nodes were removed to improve accuracy.
* What were some problems with the initial architecture?
   Initially a 92% accuracy was obtained, to try to imposed more nonlinearity in the model the activation functions were replaced by sigmoid and a tanh. this improve the accuracy in a little more than 2%. Then the maxpool node in the second layer was removed and the accuracy got improved again up to ~96%
* Why did you believe it would be relevant to the traffic sign application?
  The good results obtained in hand writting recognition made good reasons to start working with this architecture.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  The final net behaving similar in training and validation was an indicative of a well working network.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs (plus one) that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4]

![alt text][image5] ![alt text][image6]![alt text][image7]

The first image might be difficult to classify because there are very similar trafic signs in a triangular shape with a lot of different information whithin, so the net might confuse each other.
The second image was a little inclined.
The third share some similarities to the sixth sign.
The fourth signal was low in contrast and lacked of color, this could imply some dificulties.
And the fith image might look simple in principle, but the fact it have a sticker in the bottom migth imply some "confussion" to the net.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the sevententh cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection     | Right-of-way at the next intersection   				| 
| No passing     				| No Passing					|
| Speed Limit (80Km/h)			| Speed Limit (80Km/h)			|
| End of all speed and passing limits   			| End of all speed and passing limits				|
| No Entry				| No Entry					|
| Speed Limit (30Km/h)			| Speed Limit (30Km/h)			|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.6%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the first image, the model is quite sure that this is a Right-of-way sign (probability of 0.999797), and the image does contain a Right-of-way sign. The top five soft max probabilities were:

| Probability         	|     Prediction						   | 
|:-----------:|:---------------------------------------------------------------:|
| .999797	 | Right-of-way							   |
| 3.79e-5   | No Entry						   		   |
| 3.41e-5	 | Priority Road							   |
| 3.41e-5   | Double Curve					 		   |
| 3.37e-6	 | Traffic Signal  							   |


For the second image the result were quite conclusive as well:

| Probability         	|     Prediction						  | 
|:-----------:|:---------------------------------------------------------------:|
| .999855	 | No Passing								   |
| 3.95e-5   | vehicles over 3.5 metric tons prohibited		   |
| 2.71e-5	 | No Passing for vehicles over 3.5 metric tons	   |
| 1.55e-5	 | Priority Road						   	   |
| 1.03e-5	 | Yield					 		   		   |

Similar results for the third image:

| Probability         	|     Prediction						  | 
|:-----------:|:---------------------------------------------------------------:|
| .920753	 | Speed Limit (80Km/h)						   |
| .044312  | Speed Limit (30Km/h)						   |
| .029427	 | Speed Limit (50Km/h)						   |
| .002774	 | Speed Limit (70Km/h)				 		   |
| 000826	 | Speed Limit (60Km/h)						   |
 
The fourth is not that certain, even though is three times the next posibilty:
 
| Probability         	|     Prediction						  | 
|:-----------:|:---------------------------------------------------------------:|
| .758075	 | End of all speed and passing limits			   |
| .226955  | End of speed limit						   |
| .008345	 | Speed Limit								   |
| .001893	 | End of no Passing				 		   |
| .001084	 | Speed Limit (20Km/h)						   |

The Fifth virtually have only one option and the rest is only noise:

| Probability         	|     Prediction						  | 
|:-----------:|:---------------------------------------------------------------:|
| .999821	 | No Entry								   |
| .000172  | Speed Limit (120Km/h)					   |
| 1.36e-6	 | Speed Limit (20Km/h)						   |
| 1.14e-6	 | Bumpy Road					 		   |
| 8.61e-7	 | Bicycles crossing							   |

And finally the sixth

| Probability         	|     Prediction						  | 
|:-----------:|:---------------------------------------------------------------:|
| .997353	 | Speed Limit (30Km/h)						   |
| .001411   | Speed Limit (80Km/h)					 	   |
| .000510	 | Speed Limit (70Km/h)						   |
| .000282	 | Speed Limit (50Km/h)				 		   |
| .000206	 | Speed Limit (20Km/h)						   |
