# ImageColorization
The SoC 2021 project.

## Useful Links
A Machine Learning course [here](https://www.coursera.org/learn/machine-learning)  
Notes on Machine Learning [here](http://cs229.stanford.edu/summer2019/cs229-notes1.pdf)

## Linear Regression
The main aim is to estimate a linear equation representing the given set of data. There are two approaches to this.   
1. A closed form solution.  
   This can be directly obtained by solving the linear differential equation.
2. An iterative approach.  
    This is similar to **Gradient Descent**. We try to obtain the minima (L1, L2 norm etc) by calculating the gradient at each point and moving in small steps along the gradient vector. 
Refer to [this](https://youtu.be/8PJ24SrQqy8) video for more details. 
    
## Logistic Regression
Refer to the following [link](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc) to see an example of logistic regression.

## Gradient Descent
[Here](https://youtu.be/sDv4f4s2SB8) is a useful video.  
An article about Gradient Descent [here](https://medium.com/@lachlanmiller_52885/machine-learning-week-1-cost-function-gradient-descent-and-univariate-linear-regression-8f5fe69815fd)  
A useful post on GeeksForGeeks [here](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/)  

## Deep Learning
A book on deep learning [here](http://neuralnetworksanddeeplearning.com/index.html)

### Chapter 1
**Perceptrons**  
So how do perceptrons work? A perceptron takes several binary inputs, x1,x2,…, and produces a single binary output.
![image](https://user-images.githubusercontent.com/52414199/118229393-8ba12a80-b4a9-11eb-8905-58ccb2eaaad9.png)  
A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence. By varying the weights and the threshold, we can get different models of decision-making. Using the bias instead of the threshold, the perceptron rule can be rewritten:
![image](https://user-images.githubusercontent.com/52414199/118229509-bd19f600-b4a9-11eb-83bd-e3667417de98.png)  
Another way perceptrons can be used is to compute the elementary logical functions we usually think of as underlying computation, functions such as AND, OR, and NAND.  

**Sigmoid Neurons**
A small change in the weights or bias of any single perceptron in the network can sometimes cause the output of that perceptron to completely flip, say from 0 to 1. We can overcome this problem by introducing a new type of artificial neuron called a sigmoid neuron. Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output. That's the crucial fact which will allow a network of sigmoid neurons to learn.  
Just like a perceptron, the sigmoid neuron has inputs, x1,x2,…. But instead of being just 0 or 1, these inputs can also take on any values between 0 and 1. So, for instance, 0.638… is a valid input for a sigmoid neuron. Also just like a perceptron, the sigmoid neuron has weights for each input, w1,w2,…, and an overall bias, b. But the output is not 0 or 1. Instead, it's σ(w⋅x+b), where σ is called the sigmoid function.  
To understand the similarity to the perceptron model, suppose z≡w⋅x+b is a large positive number. Then e−z≈0 and so σ(z)≈1. In other words, when z=w⋅x+b is large and positive, the output from the sigmoid neuron is approximately 1, just as it would have been for a perceptron. Suppose on the other hand that z=w⋅x+b is very negative. Then e−z→∞, and σ(z)≈0. So when z=w⋅x+b is very negative, the behaviour of a sigmoid neuron also closely approximates a perceptron.  

**Neural Networks Architecture**
A typical neural network consists of an input layer, an output layer and 0 or more hidden layers. *Hidden Layers* are just layers which are neither the input layer nor the output layer. Each layer consists neurons whose input is taken from the previous layer and the output acts like the input to the next layer.  
Each layer in the neural network captures an abstraction/feature of the object/data we are trying to learn. For example, the digit recognition neural network has 3 layers. The input layer consists of 784 (28 /* 28) neurons and the output layer has 10 neurons each corresponding to an identity of a digit. The hidden layer has 20 neurons. One may think of the hidden layer as capturing the main features of a digit such as loops, curves and straight lines.  
The aim of a neural network is to learn the input data and output the corresponding label to a particular input. To do this, we define a **cost function**. For example, in the digit recognition neural network, we use a _least squared error_ norm to minimise the error in the output. This function is particularly useful because it is convex and it is minimized when the output of the neural network matches the correct output.  
![image](https://user-images.githubusercontent.com/52414199/118230609-91980b00-b4ab-11eb-8107-85bd0077eab3.png)  

To minimise the cost function, we use a calculus approach. Notice that the above cost function is a function of all the weights and biases in the network. We aim to minimise the cost function for a given input dataset by tuning the weights and biases in the network. We tweak each of these variables (weights and biases) and measure the change in the cost function for *all* the inputs. This way, we identify the **gradient of the function**. We move in the negative direction of the gradient to minimise the cost function. We repeat this until convergence (A threshold accuracy). This algorithm is called as **Gradient Descent**.  
![image](https://user-images.githubusercontent.com/52414199/118231313-97dab700-b4ac-11eb-858b-480751a2645e.png)  
Here, $\eta$ refers to the **learning rate** and is one of the _hyperparameters_.

In the above algorithm, we calculated the cost function for all the inputs in each iteration. This turns out to be a computationally expensive step. Therefore, we select a **mini-batch** from the input and evaluate the cost function over this input. We assume that this cost function is a representative of the real cost function. We change the mini-batch in each iteration so as to cover all the inputs. The change in the cost function looks like this:
