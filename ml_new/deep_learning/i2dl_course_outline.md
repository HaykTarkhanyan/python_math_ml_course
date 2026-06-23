# `lecture_i2dl` - Granular Course Outline (slide by slide)

A frame-by-frame outline of the **Introduction to Deep Learning (I2DL)** slides in
`_reference/lecture_i2dl/`. This is the detailed companion to `i2dl_slides_summary.md`
(which is the deck-level overview). Generated: 2026-06-22.

**How to read this**

- Decks are ordered along the course arc (intro -> MLPs -> optimization -> CNNs -> RNNs -> autoencoders -> generative models -> adversarial).
- Each `###` heading is one deck: its chapter title, then ``folder/file.tex`` and the frame count.
- Numbered items are the **logical frames** in presentation order. `vbframe` continuations that share a title are merged into one entry; a blank continuation slide shows as `(cont.)`.
- Sub-bullets are short gists pulled from the slide's own bullet text (first few points). Math/formulas are omitted for readability, so some gists read as prose with the equations dropped. For exact formulas, open the deck PDF under `slides-pdf/<folder>/`.
- "Frames" = `\begin{frame}` / `\begin{vbframe}` blocks; the rendered PDF usually has more pages because `vbframe` auto-splits long content.

---


## intro/  -  Introduction and history

### Introduction
`intro/slides-intro-introduction.tex` - 9 frames

1. **What is Deep Learning**
   - Deep learning is a subfield of ML based on artificial neural networks
2. **Deep Learning and Neural Networks**
   - Deep learning itself is not new
   - Neural networks have been around since the 70s
   - Deep neural networks, i.e., networks with multiple hidden layers, are not much younger
3. **Image Classification with Neural Networks**
4. **Possible use-cases**
   - It is high dimensional
   - Each single feature itself is not very informative but only a combination of them might be
   - There is a large amount of training data
5. **Possible use-case: Images**
   - High Dimensional: A color image with (3 Colors) pixels already has features
   - Informative: A single pixel is not meaningful in itself
   - Training Data: Depending on applications huge amounts of data are available
6. **Possible use-case: Text**
   - High Dimensional: Each word can be a single feature (300000 words in the German language)
   - Informative: A single word does not provide much context
   - Training Data: Huge amounts of text data available
7. **Possible use-case: Text Classification**
8. **Possible use-case: Text**
9. **Applications of Deep Learning: Speech**

### Brief History
`intro/slides-intro-brief-history.tex` - 2 frames

1. **A brief history of neural networks**
   - 1943: The first artificial neuron, the "Threshold Logic Unit (TLU)", was proposed by Warren McCulloch & Walter Pitts
   - The model is limited to binary inputs
   - The weight are not adjustable, so learning could only be achieved by changing the threshold
2. **What the field looks like today**
   - A major shift in the 2020s was the move from highly specialized models to foundation models: large pretrained models that can be adapted to many ...
   - The frontier is increasingly multimodal: models can jointly process text, images, audio, video, and actions
   - Modern systems are also becoming more interactive: they can write code, search, retrieve documents, and call external tools


## mlps/  -  From a single neuron to deep feed-forward nets

### Single Neuron / Perceptron
`mlps/slides-mlps-single-neuron.tex` - 3 frames

1. **A Single Neuron**
   - The perceptron is a single artificial neuron and the basic computational unit of neural networks
   - It is a weighted sum of input values, transformed by
   - The identity function gives us the simple linear regression
2. **A Single Neuron: Hypothesis Space**
   - The hypothesis space that is formed by single neuron
   - If is the logistic sigmoid or identity function, corresponds to the hpothesis space of logistic or linear regression, respectively
3. **A Single Neuron: Optimization**
   - To optimize this model, we minimize the empirical risk
   - For regression, we typically use the L2 loss (rarely L1)
   - For binary classification, we typically apply the cross entropy loss (also known as Bernoulli loss)

### Single hidden layer neural networks
`mlps/slides-mlps-single-hidden-layer-networks.tex` - 5 frames

1. **Motivation**
   - The graphical way of representing simple functions/models, like logistic regression. Why is that useful?
   - Because individual neurons can be used as building blocks of more complicated functions
   - Networks of neurons can represent extremely complex hypothesis spaces
2. **Representation Learning**
   - Before deep learning took off, features for tasks like machine vision and speech recognition were hand-designed by domain experts. This step of the ...
3. **Single Hidden Layer Networks**
   - Affine Transformation: a weighted sum of inputs plus bias
   - Activation: a non-linear transformation on the weighted sum
   - Hidden Layer: having a set of neurons
4. **Single Hidden Layer Networks: Example**
5. **Hidden Layer: Activation Function**
   - If the hidden layer does not have a non-linear activation, the network can only learn linear decision boundaries
   - A lot of different activation functions exist
   - Currently the most popular choice is the ReLU (rectified linear unit)

### Single hidden layer neural networks
`mlps/slides-mlps-mlps-as-predictor.tex` - 5 frames

- Duplicate of `mlps/slides-mlps-single-hidden-layer-networks.tex` (identical frames); see above.

### MLP -- Matrix Notation
`mlps/slides-mlps-matrix-notation.tex` - 2 frames

1. **Single Hidden Layer Networks: Notations**
   - The input is a column vector with dimensions
   - is a weight matrix with dimensions, where is the amount of hidden neurons
   - For example, to obtain, we pick the first column of
2. **Single Hidden Layer Networks: Notation**
   - The network has hidden neurons with
   - Vectorized notation
   - where the (hidden layer) activation function is applied element-wise to

### MLP -- Multi-Layer Feedforward Neural Networks
`mlps/slides-mlps-multilayer-FNNs.tex` - 5 frames

1. **Feedforward neural networks**
   - We will now extend the model class once again, such that we allow an arbitrary amount of hidden layers
   - The general term for this model class is (multi-layer) feedforward networks (inputs are passed through the network from left to right, no ...
   - We can characterize those models by the following chain structure: where and are the activation function and the weighted sum of hidden layer ...
2. **Feedforward neural networks: Example**
3. **Why add more layers?**
   - Multiple layers allow for the extraction of more and more abstract
   - Each layer in a feed-forward neural network adds its own degree of non-lnearity to the model
4. **Deep neural networks**
   - Training DNNs on CPUs was too slow to be practical. Switching over to GPUs cut down training time by more than an order of magnitude
   - When dataset sizes are small, other models (such as SVMs) and techniques (such as feature engineering) often outperform them
   - The availability of large datasets and novel architectures that are capable of handling even complex tensor-shaped data (e.g. CNNs for image data) ...
5. **(untitled)**

### Single Hidden Layer Networks for Multi-Class Classification
`mlps/slides-mlps-multiclass-classification.tex` - 3 frames

1. **Multi-class Classification**
   - We have only considered regression and binary classification problems so far
   - How can we get a neural network to perform multiclass classification?
   - The first step is to add additional neurons to the output layer
2. **Multi-class Classification: Example**
3. **Optimization: Softmax Loss**
   - The loss function for a softmax classifier is
   - This is equivalent to the cross-entropy loss when the label vector is one-hot coded (e.g. )
   - Optimization: Again, there is no analytic solution

### Universal Approximation
`mlps/slides-mlp-univ-approx-theorem.tex` - 4 frames

1. **Universal approximation property**
   - This means that for a given target function there exists a
   - Usually, as the networks come closer and closer to, they
   - A network with fixed layer sizes can only model a subspace of all
2. **Example: Regression/Classification**
   - Let's look at a few examples of the types of functions and decisions boundaries learnt by neural networks (with a single hidden layer) of various ...
   - "size" here refers to the number of neurons in the hidden layer
   - The number of "iterations" in the following slides corresponds to the number of steps of the applied iterative optimization algorithm (stochastic ...
3. **Regression Ex.: 1000 training iterations**
4. **Classification: 500 training iterations**

### XOR-Problem
`mlps/slides-mlps-xor.tex` - 2 frames

1. **Example: XOR Problem**
   - Suppose we have four data points
   - The XOR gate (exclusive or) returns true, when an odd number of inputs are true
   - Can you learn the target function with a logistic regression model?
2. **Neural Networks: Optimization**
   - In this simple example we actually guessed the values of the parameters for,, and
   - That won't work for more sophisticated problems!
   - We will learn later about iterative optimization algorithms for automatically adapting weights and biases


## opt1/  -  Backpropagation, training, hardware/software

### Chain Rule and Computational Graphs
`opt1/slides-opt1-comp-graphs.tex` - 5 frames

1. **Chain rule of calculus**
   - The chain rule can be used to compute derivatives of the composition of two or more functions
   - If and, the chain rule yields
2. **Computational graphs**
   - CGs are nested expresssions, visualized as graphs
   - Each node is a variable, either an input or derived
   - Derived variables are functions applied to other variables
3. **Chain rule of calculus: Example 1**
   - Suppose we have the following computational graph
   - To compute the derivative of
4. **Chain rule of calculus: Example 2**
5. **Computational Graph: Neural Net**

### Basic Backpropagation 1
`opt1/slides-opt1-basic-backpropagation1.tex` - 5 frames

1. **Backpropagation: Basic Idea**
   - Forward pass: Inputs flow through model to outputs
   - Backward pass: Loss flows backwards to update weights so error is reduced
2. **XOR example**
   - As activations (hidden and outputs) we use the logistic
   - We run one FP and BP on with
   - We use L2 loss between 0-1 labels and the predicted probabilities
3. **Forward pass**
   - We will divide the FP into four steps
   - the inputs of
   - the activations of
4. **Backward pass**
   - to reuse the results of the forward pass (here: )
   - reuse the violetintermediate results from the chain rule
   - the derivative of the activations and some affine functions
5. **Result**
   - We can do this for all weights
   - Yields and loss
   - Before, we had and higher loss

### Basic Backpropagation 2
`opt1/slides-opt1-basic-backpropagation2.tex` - 3 frames

1. **Backward Computation and Caching**
   - Examining the two expressions
   - Significant overlap / redundancy in the two expressions
   - Again: Let's call this subexpression and cache it
2. **Backprop: Recursion**
   - Let us now derive a general formulation of backprop
   - The neurons in layers, and are indexed by, and, respectively
   - The output layer will be referred to as layer O
3. **(untitled)**

### Basic Training
`opt1/slides-opt1-basic-training.tex` - 8 frames

1. **Training Neural Networks**
   - In DL, represents the weights (and biases) of the NN
   - Often, L2 in regression
   - or cross-entropy for binary classification
2. **Gradient descent**
   - Neg. risk gradient points in the direction of the steepest descent
   - Standing at a point, we locally improve by
   - is called step size or learning rate
3. **Gradient Descent and Optimality**
   - GD is a greedy algorithm: In every iteration, it makes locally optimal moves
   - If is convex and differentiable, and its gradient is Lipschitz continuous, GD is guaranteed to converge to the global minimum (for small enough ...
   - However, if has multiple local optima and/or saddle points, GD might only converge to a stationary point (other than the global optimum), depending ...
4. **Learning rate**
5. **Learning Rate**
6. **Weight Initialization**
   - Weights (and biases) of an NN must be initialized in GD
   - We somehow must "break symmetry" -- which would happen in full-0-initialization
   - Weights are typically drawn from a uniform a Gaussian distribution (both centered at 0 with a small variance)
7. **Stochastic gradient descent**
   - Using the entire training set in GD to is called batch or deterministic or offline training. This can be computationally costly or impossible, if ...
   - Idea: Instead of letting the sum run over the whole dataset, use small stochastic subsets (minibatches), or only a single
   - If batches are uniformly sampled from, our stochastic gradient is in expectation the batch gradient
8. **(untitled)**

### Hardware and Software
`opt1/slides-opt1-hardware-and-software.tex` - 5 frames

1. **Hardware for Deep Learning**
   - Deep neural networks require special hardware to be trained efficiently
   - The training is done using Graphics Processing Units (GPUs) and a special programming language called CUDA
   - Training on standard CPUs takes a very long time
2. **Graphics Processing Units (GPUs)**
   - Initially developed to accelerate the creation of graphics
   - Massively parallel: identical and independent computations for every pixel
   - Computer Graphics makes heavy use of linear algebra (just like neural networks)
3. **Tensor Processing Units (TPUs)**
   - Specialized and proprietary chip for deep learning developed by Google
   - Hundreds of teraFLOPS per chip
   - Can be connected together in pods of thousands TPUs each (result: hundreds of petaFLOPS per pod)
4. **And everything else**
   - With such powerful devices, memory/disk access during training become the bottleneck
   - Nvidia DGX-1: Specialized solution with eight Tesla V100 GPUs, dual Intel Xeon, 512 GB of RAM, 4 SSD disks of 2TB each
   - Specialized hardware for on-device inference
5. **Software for Deep Learning**
   - CUDA is a very low level programming language and thus writing code for deep learning requires a lot of work
   - Deep learning (software) frameworks
   - Abstract the hardware (same code for CPU/GPU/TPU)


## regu/  -  Regularization

### Basic Regularization
`regu/slides-regu-basic-regularization.tex` - 8 frames

1. **Regularization**
   - Any technique that is designed to reduce the test error possibly at the expense of increased training error can be considered a form of regularization
   - Regularization is important in DL because NNs can have extremely high capacity (millions of parameters) and are thus prone to overfitting
2. **Revision: Regularized Risk Minimization**
   - The goal of regularized risk minimization is to penalize the complexity of the model to minimize the chances of overfitting
   - By adding a parameter norm penalty term \(J\) to the empirical risk we obtain a regularized cost function
   - Therefore, instead of pure empirical risk minimization, we add a penalty
3. **L2-regularization / Weight decay**
4. **Equivalence to Constrained Optimization**
5. **Example: Weight decay**
   - We fit the huge neural network on the right side on a smaller fraction of MNIST (5000 train and 1000 test observations)
   - Weight decay
6. **Tensorflow Playground**
7. **Tensorflow Playground - Exercise**
8. **(untitled)**

### Early Stopping
`regu/slides-regu-early-stopping.tex` - 3 frames

1. **Early Stopping**
   - When training with an iterative optimizer such as SGD, it is commonly the case that, after a certain number of iterations, generalization error ...
   - Early stopping refers to stopping the algorithm early before the generalization error increases
   - Split training data into and (e.g. with a ratio of 2:1)
2. **Early Stopping and**
   - For simple case of LM with squared loss and GD optim initialized at: Early stopping has exact correspondence with regularization/WD
   - Small ( regu. ) large (complexity ) and vice versa
3. **SGD Trajectory and**

### Dropout and Augmentation
`regu/slides-regu-ensemble-dropout-augmentation.tex` - 8 frames

1. **Recap: Ensemble Methods**
   - Idea: Train several models separately, and average their prediction (i.e. perform model averaging)
   - Intuition: This improves performance on test set, since different models will not make the same errors
   - Ensembles can be constructed in different ways, e.g
2. **Dropout**
   - Method: during training, random subsets of the neurons are removed from the network (they are "dropped out").This is done by artificially setting the ...
   - Whether a given unit/neuron is dropped out or not is completely independent of the other units
   - If the network has (input/hidden) units, applying dropout to these units can result in possible 'subnetworks'
3. **Dropout: Algorithm**
   - To train with dropout a minibatch-based learning algorithm such as stochastic gradient descent is used
   - For each training case in a minibatch, we randomly sample a binary vector/mask with one entry for each input or hidden unit in the network. The ...
   - The probability of sampling a mask value of 0 (dropout) for one unit is a hyperparameter known as the 'dropout rate'
4. **Dropout: Weight scaling**
   - The weights of the network will be larger than normal because of dropout. Therefore, to obtain a prediction at test time the weights must be first ...
   - This means that if a unit (neuron) is retrained with probability during training, the weight at test time of that unit is multiplied by
   - Weight scaling ensures that the expected total input to a neuron/unit at test time is roughly the same as the expected total input to that unit at ...
5. **Dropout: Example**
   - To demonstrate how dropout can easily improve generalization we compute neural networks with the structure showed on the right
   - Each neural network we fit has different dropout probabilities, a tuple where one probability is for the input layer and one is for the hidden ...
6. **Dropout, weight decay or both?**
7. **Dataset Augmentation**
   - Problem: low generalization because high ratio of
   - Idea: artificially increase the train data
   - Limited data supply create fake data!
8. **(untitled)**


## opt2/  -  Optimization challenges, advanced optimizers, activations, init

### Challenges in Optimization
`opt2/slides-optim-challenges.tex` - 16 frames

1. **Challenges in Optimization**
   - In this section, we summarize several of the most prominent challenges regarding training of deep neural networks
   - Traditionally, machine learning ensures that the optimization problem is convex by carefully designing
   - Furthermore, we will see in this section that even convex optimization is not without its complications
2. **Effects of curvature**
3. **Second derivative and curvature**
   - The second derivative corresponds to the curvature of the graph of a function
   - The Hessian matrix of a function is the matrix of second-order partial derivatives
   - The second derivative in a direction
4. **Ill-conditioned Hessian matrix**
5. **Curvature and Step-size in GD**
   - Let us consider the second-order Taylor approximation as a local approximation of the of around a current point (with gradient )
   - Furthermore, Taylor's theorem states (proof in Koenigsberger (1997), p. 68)
   - One GD step with a learning rate yields new parameters and a new approximated loss value
6. **Ill-conditioning**
   - GD is unaware of large differences in curvature, and can only walk into the direction of the gradient
   - Choosing a too large step-size will then cause the descent direction change frequently (jumping around)
   - needs to be small enough, which results in a low progress
7. **Unimodal vs. Multimodal loss surfaces**
8. **Multimodal function**
9. **Only locally optimal moves**
   - If the training algorithm makes only locally optimal moves (as in gradient descent), it may move away from regions of much lower cost
   - In the figure above, initializing the parameter on the "wrong" side of the hill will result in suboptimal performance
   - In higher dimensions, however, it may be possible for gradient descent to go around the hill but such a trajectory might be very long and result in ...
10. **Local minima**
   - If we swap incoming weight vectors for neuron and
   - with hidden units and one hidden layer
   - If we multiply incoming weights of a ReLU neuron with
11. **Saddle Points**
   - In optimization we look for areas with zero gradient
   - A variant of zero gradient areas are saddle points
   - For the empirical risk of a neural network, the expected ratio of the number of saddle points to local minima typically grows exponentially with
12. **Saddle Points: Example**
   - Along, the function curves upwards (eigenvector of the Hessian with positive eigenvalue). Along, the function curves downwards (eigenvector of the ...
13. **Saddle Points**
   - So how do saddle points impair optimization?
   - First-order algorithms that use only gradient information might get stuck in saddle points
   - Second-order algorithms experience even greater problems when dealing with saddle points. Newtons method for example actively searches for a region ...
14. **Cliffs and exploding gradients**
   - As a result from the multiplication of several parameters, the emprirical risk for highly nonlinear deep neural networks often contain sharp ...
   - That may result in very high derivatives in some places
   - As the parameters get close to such cliff regions, a gradient descent update can catapult the parameters very far
15. **Example: cliffs and exploding gradients**
16. **(untitled)**

### Advanced Optimization
`opt2/slides-advanced-optim.tex` - 22 frames

1. **Momentum**
   - While SGD remains a popular optimization strategy, learning with it can sometimes be slow
   - Momentum is designed to accelerate learning, especially when facing high curvature, small but consistent or noisy gradients
   - Momentum accumulates an exponentially decaying moving average of past gradients
2. **Momentum: Example**
3. **Momentum: Illustration**
4. **SGD with momentum**
5. **SGD with and without momentum**
6. **Momentum in practice**
   - Lets try out different values of momentum (with SGD) on the MNIST data
   - We apply the same architecture we have used a dozen of times already (note that we used in all computations so far, i.e. in chapter 1 and 2)!
7. **Nesterov momentum**
   - Momentum aims to solve poor conditioning of the Hessian but also variance in the stochastic gradient
   - Nesterov momentum modifies the algorithm such that the gradient is evaluated after the current velocity is applied
   - We can interpret Nesterov momentum as an attempt to add a correction factor to the basic method
8. **SGD with Nesterov Momentum**
9. **Momentum vs. Nesterov Momentum**
10. **Learning Rate**
   - The learning rate is a very important hyperparameter
   - To systematically find a good learning rate, we can start at a very low learning rate and gradually increase it (linearly or exponentially) after ...
   - We can then plot the learning rate and the training loss for each batch
11. **Learning Rate Schedule**
   - We would like to force convergence until reaching a local minimum
   - Applying SGD, we have to decrease the learning rate over time, thus (learning rate at training iteration )
   - The estimator is computed based on small batches
12. **Cyclical Learning Rates**
   - Another option is to have a learning rate that periodically varies according to some cyclic function
   - Therefore, if training does not improve the loss anymore (possibly due to saddle points), increasing the learning rate makes it possible to rapidly ...
   - Recall, saddle points are far more likely than local minima in deep nets
13. **Adaptive learning rates**
   - The learning rate is reliably one of the hyperparameters that is the most difficult to set because it has a significant impact on the models ...
   - Naturally, it might make sense to use a different learning rate for each parameter, and automatically adapt them throughout the training process
14. **Adagrad**
   - Adagrad adapts the learning rate to the parameters
   - In fact, Adagrad scales learning rates inversely proportional to the square root of the sum of the past squared derivatives
   - Parameters with large partial derivatives of the loss obtain a rapid decrease in their learning rate
15. **RMSProp**
   - RMSprop is a modification of Adagrad
   - It's intention is to resolve Adagrad's radically diminishing learning rates
   - The gradient accumulation is replaced by an exponentially weighted moving average
16. **Adam**
   - Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter
   - Adam uses the first and the second moments of the gradients
   - Adam keeps an exponentially decaying average of past gradients (first moment)
17. **Batch Normalization**
   - Batch Normalization (BatchNorm) is an extremely popular technique that improves the training speed and stability of deep neural nets
   - It is an extra component that can be placed between each layer of the neural network
   - It works by changing the "distribution" of activations at each hidden layer of the network
18. **Batch Normalization: Illustration**
   - So far, we have applied batch-norm to the activation. It is possible (and more common) to apply batch norm to before passing it to the nonlinear ...
19. **Batch Normalization**
   - The key impact of BatchNorm on the training process is this: It reparametrizes the underlying optimization problem to make its landscape ...
   - One aspect of this is that the loss changes at a smaller rate and the magnitudes of the gradients are also smaller (see Santurkar et al. 2018)
20. **Batch Normalization: Prediction**
   - Once the network has been trained, how can we generate a prediction for a single input (either at test time or in production)?
   - One option is to feed the entire training set to the (trained) network and compute the means and standard deviations
   - More commonly, during training, an exponentially weighted running average of each of these statistics over the minibatches is maintained
21. **Batch Normalization**
   - For our final benchmark in this chapter we compute two models to predict the mnist data
   - One will extend our basic architecture such that we add batch normalization to all hidden layers
   - We use SGD as optimizer with a momentum of 0.9, a learning rate of 0.03 and weight decay of 0.001
22. **(untitled)**

### Modern Activation Functions
`opt2/slides-activations.tex` - 7 frames

1. **Hidden activations**
   - Recall, hidden-layer activation functions make it possible for deep neural nets to learn complex non-linear functions
   - The design of hidden units is an extremely active area of research
   - In the following, we will limit ourselves to the most popular activations - Sigmoidal activation and ReLU
2. **Sigmoidal activations**
   - Sigmoidal functions such as tanh and the logistic sigmoid bound the outputs to a certain range by "squashing" their inputs
   - In each case, the function is only sensitive to its inputs in a small neighborhood around
   - Furthermore, the derivative is never greater than 1 and is close to zero across much of the domain
3. **Sigmoidal Activation Functions**
   - Saturating Neurons
   - We know: for
   - Neurons with sigmoidal activations "saturate" easily, that is, they stop being responsive when
4. **Rectified Linear Units (ReLU)**
   - The ReLU activation solves the vanishing gradient problem
   - In regions where the activation is positive, the derivative is
   - Derivatives do not vanish along paths containing "active" neurons
5. **Generalizations of ReLU**
   - There exist several generalizations of the ReLU activation that have non-zero derivatives throughout their domains
   - Unlike the ReLU, when the input to the Leaky ReLU activation is negative, the derivative is which is a small positive value
   - A variant of the Leaky ReLU is the Parametric ReLU (PReLU) which learns the from the data through backpropagation
6. **Output Activations**
   - As we have seen previously, the role of the output activation is to get the final score on the same scale as the target
   - The output activations and the loss functions used to train neural networks can be viewed through the lens of maximum likelihood estimation (MLE)
   - In general, the function represented by the neural network defines the conditional in a supervised learning task
7. **(untitled)**

### Network Initializations
`opt2/slides-initialization.tex` - 5 frames

1. **Practical Initialization**
   - The weights (and biases) of a neural network must be assigned some initial values before training can begin
   - The choice of the initial weights (and biases) is crucial as it determines whether an optimization algorithm converges, how fast and whether to a ...
   - Initialization strategies to achieve "nice" properties are difficult to find, because there is no good understanding which properties are preserved ...
2. **Weight Initialization**
   - It is important to initialize the weights randomly in order to "break symmetry". If two neurons (with the same activation function in a fully ...
   - Furthermore, the initial weights should not be too large, because this might result in an explosion of weights or high sensitivity to changes in the ...
   - Weights are typically drawn from a uniform distribution or a Gaussian centered at 0 with a small variance
3. **Weight initialization: Example**
   - We use a spiral planar data set to compare the following strategies
   - For each strategy, a neural network with one hidden layer with 100 units, ReLU activation and Gradient Descent as optimizer was used
4. **Bias initialization**
   - Typically, we set the biases for each unit to heuristically chosen constants
   - Setting the biases to zero is compatible with most weight
   - However, deviations from 0 can be made individually, for example, in order to obtain the right marginal statistics of the output unit or to avoid ...
5. **(untitled)**


## cnn1/  -  CNN fundamentals

### CNN: Introduction
`cnn1/slides-cnn-introduction.tex` - 4 frames

1. **Convolutional Neural Networks**
   - Convolutional Neural Networks (CNN, or ConvNet) are a powerful family of neural networks that are inspired by biological processes in which the ...
   - Since 2012, given their success in the ILSVRC competition, CNNs are popular in many fields
   - Common applications of CNN-based architectures in computer vision are
2. **CNNs - What for?**
3. **CNNs - A First Glimpse**
   - Input layer takes input data (e.g. image, audio)
   - Convolution layers extract feature maps from the previous layers
   - Pooling layers reduce the dimensionality of feature maps and filter meaningful features
4. **(untitled)**

### Convolutional Operation
`cnn1/slides-cnn-conv2d.tex` - 7 frames

1. **Filters to extract features**
   - Filters are widely applied in Computer Vision (CV) since the 70's
   - One prominent example: Sobel-Filter
   - It detects edges in images
2. **Horizontal vs Vertical edges**
3. **Filters to extract features**
   - Let's do this on a dummy image
   - How to represent a digital image?
   - Basically as an array of integers
4. **Why do we need to know all of that?**
   - What we just did was extracting pre-defined features from our input (i.e. edges)
   - A convolutional neural network does almost exactly the same: extracting features from the input
   - The main difference is that we usually do not tell the CNN what to look for (pre-define them), the CNN decides itself
5. **Working with images**
   - In order to understand the functionality of CNNs, we have to familiarize ourselves with some properties of images
   - Grey scale images
   - Matrix with dimensions height width 1
6. **The 2d convolution**
   - Suppose we have an input with entries (think of pixel values)
   - The filter we would like to apply has weights
   - To obtain we simply compute the dot product
7. **(untitled)**

### Convolutions - Mathematical Perspective
`cnn1/slides-cnn-math.tex` - 8 frames

1. **Convolutions: A Deeper Look**
   - CNNs borrow their name from a mathematical operation termed convolution that originates in Signal Processing
   - Basic understanding of this concept and related operations improves the understanding of the CNN functionality
   - Intuition 1: weighted smoothing of with weighting function
2. **1D Convolution Animation**
3. **Properties of the Convolution**
   - Commutativity
   - Associativity
   - Distributivity
4. **Related operations**
   - Convolution is strongly related to two other mathematical operators
   - Fourier transform via the Convolution Theorem
   - Cross correlation
5. **Convolution Theorem**
   - Fourier transform of the convolution of two functions can be expressed as the product of their Fourier transforms
   - Transformation of a signal from time to frequency domain
   - Convolution in the time domain is equivalent to multiplication in frequency domain
6. **Convolution Theorem - Proof**
7. **Cross Correlation**
   - Measurement for similarity of two functions
   - More specifically, at which position are the two functions most similar to each other? Where does the pattern of match the best?
   - Slide with over and at each discrete step compute the sum of the product of their elements
8. **Cross Correlation vs. Convolution**
   - Cross correlation is not commutative
   - but often implemented instead of convolution in the practice

### Properties of Convolution
`cnn1/slides-cnn-properties-of-convolution.tex` - 3 frames

1. **Sparse interactions**
   - What does that mean?
   - Our CNN has a receptive field of 4 neurons
   - That means, we apply a local search for features
2. **Sparse Connections and Parameter sharing**
   - Why is that good?
   - Less parameters drastically reduce memory requirements
   - Faster runtime
3. **Nonlinearity in feature maps**
   - As in dense nets, we use activation functions on all feature map entries to introduce nonlinearity in the net
   - Typically rectified linear units (ReLU) are used in CNNs
   - They reduce the danger of saturating gradients compared to sigmoid activations

### CNN Components
`cnn1/slides-cnn-components.tex` - 9 frames

1. **Input Channel**
   - An image consists of the smallest indivisible segments called pixels with a strength often known as the pixel intensity
   - A grayscale image has a single input channel with the value of each pixel representing the amount of light/brightness in the pixel
   - A grayscale value can lie between 0 and 255 (0 = black, 255 = white)
2. **(untitled)**
   - A colored digital image usually comes with three color channels, i.e. the Red-Green-Blue channels, popularly known as the RGB values
   - Each pixel can be represented by a vector of three numbers (each ranging from 0 to 255) for the three primary color channels
3. **Valid Padding**
   - Suppose we have an input of size and a filter of size
   - The filter is only allowed to move inside of the input space
   - That will inevitably reduce the output dimensions
4. **Same Padding**
   - Suppose the following situation: an input with dimensions and a filter with size
   - We would like to obtain an output with the same dimensions as the input
   - Hence, we apply a technique called zero padding. That is to say pad zeros around the input
5. **Padding and Network Depth**
6. **Max Pooling**
   - We've seen how convolutions work, but there is one other operation we need to understand
   - We want to downsample the feature map but optimally lose no information
   - Applying the max pooling operation, we simply look for the maximum value at each spatial location
7. **Average Pooling**
   - We've seen how max pooling worked, there are exists other pooling operation such as avg pooling, fractional pooling, LP pooling, softmax pooling ...
   - Similar to max pooling, we downsample the feature map but optimally lose no information
   - Applying the average pooling operation, we simply look for the mean/average value at each spatial location
8. **Comparison of Max and Average Pooling**
   - Avg pooling use all information by sum but max pooling use only highest value
   - In max pooling operation details are removed therefore it is suitable for sparse information (Image Classification) and avg pooling is suitable for ...
9. **(untitled)**
   - there are 3 input channel, with the size of 4x4 as an input matrices
   - one 2x2 filter (also known as kernel)
   - a single ReLu layer

### CNN: Pooling
`cnn1/slides-cnn-pooling.tex` - 3 frames

1. **Max Pooling**
   - We've seen how convolutions work, but there is one other operation we need to understand
   - We want to downsample the feature map but in the best case also lose no information
   - Applying the max pooling operation, we simply look for the maximum value at each spatial location
2. **Average Pooling**
   - We've seen how max pooling works. There also exist other pooling operations such as Avg Pooling, Fractional Pooling, LP Pooling, Wavelet Pooling ...
   - Similar to max pooling, we downsample the feature map but maybe preserve more information
   - Applying the average pooling operation, we simply look for the mean/average value at each spatial location
3. **Comparison of Max and Average Pooling**
   - Avg pooling uses the information of all inputs while Max pooling uses only the largest value
   - In Max-pooling operation details are removed, therefore it is suitable for sparse information (typically in image classification) and Avg pooling is ...

### CNN: Architecture
`cnn1/slides-cnn-architecture.tex` - 3 frames

1. **CNNs - Perspective I**
   - Schematic architecture of a CNN
   - The input tensor is convolved by different filters yielding different feature maps (coloured) in subsequent layers
   - A dense layer connects the final feature maps with the softmax-activated output neurons
2. **CNNs - Perspective II**
   - Flat view of a CNN architecture for a classification problem
   - Consists of 2 CNN layers that are each followed by max-pooling, then flattened and connected with the final output neurons via a dense layer
3. **CNNs - Perspective III**
   - Awesome interactive visualization (by Adam Harley)://scs.ryerson.ca/ aharley/vis/ here
   - Vanilla 2-layer densely-connected net on MNIST data for input digit
   - Each neuron in layer 1 is connected to each of the input neurons

### CNN Applications
`cnn1/slides-cnn-application.tex` - 9 frames

1. **Application - Image Classification**
   - One use case of CNNs is image classification
   - There exists a broad variety of network architectures for image classification such as the LeNet, AlexNet, InceptionNet and ResNet which will be ...
   - All these architectures rely on a set of sequences of convolutional layers and aim to learn the mapping from an image to a probability score over a ...
2. **CNN vs a Fully Connected net on Cifar-10**
3. **Application - Beyond Image Classification**
   - object detection
   - image captioning
   - semantic segmentation
4. **Application - Image Colorization**
   - Basic idea (introduced by Zhang et al., 2016)
   - Train the net on pairs of grayscale and colored images
   - Force it to make a prediction on the color-value for each pixel in the grayscale input image
5. **Application - Object localization**
   - Until now, we used CNNs for single-class classification of images - which object is on the image?
   - Now we extend this framework - is there an object in the image and if yes, where and which?
   - Bounding boxes can be defined by the location of the left lower corner as well as the height and width of the box: [,,, ]
6. **Semantic Segmentation**
7. **Image Captioning**
8. **Visual Question Answering**
9. **(untitled)**


## cnn2/  -  Convolution variants

### 1D / 2D / 3D Convolutions
`cnn2/slides-convolution-types.tex` - 10 frames

1. **1D Convolutions**
   - Data consists of tensors with shape [depth, xdim]
   - Depth (single-channel)
   - Univariate time series, e.g. development of a single stock price over time
2. **1D Convolutions -- Operation**
3. **1D Convolutions -- Sensor data**
4. **1D Convolutions -- Text mining**
   - 1D convolutions also have an interesting application in text mining
   - For example, they can be used to classify the sentiment of text snippets such as yelp reviews
5. **Advantages of 1D Convolutions**
   - Computational complexity: Forward propagation and backward propagation in 1D CNNs require simple array operations
   - Training is easier: Recent studies show that 1D CNNs with relatively shallow architectures are able to learn challenging tasks involving 1D signals
   - Hardware: Usually, training deep 2D CNNs requires special hardware setup (e.g. Cloud computing). However, any CPU implementation over a standard ...
6. **2D Convolutions**
7. **3D Convolutions**
   - Data consists of tensors with shape [depth, xdim, ydim, zdim]
   - Dimensions can be both temporal (e.g. video frames) or spatial (e.g. MRI)
   - Human activity recognition in video data
8. **3D Convolutions -- Data**
9. **3D Convolutions**
   - Note: 3D convolutions yield a 3D output (Jin et al., 2023)
   - Basic architecture of the CNN stays the same
   - 3D convolutions output 3D feature maps which are element-wise activated and then (eventually) pooled in 3 dimensions
10. **(untitled)**

### Important Types of Convolutions
`cnn2/slides-dilated-transposed-convolutions.tex` - 5 frames

1. **Dilated convolutions**
   - Idea: artificially increase the receptive field of the net without using more filter weights
   - The receptive field of a single neuron comprises all inputs that have an impact on this neuron
   - Neurons in the first layers capture less information of the input, while neurons in the last layers have huge receptive fields and can capture a lot ...
2. **Transposed convolutions**
   - Problem setting
   - For many applications and in many network architectures, we often want to do transformations going in the opposite direction of a normal convolution ...
   - examples include generating high-resolution images and mapping low dimensional feature map to high dimensional space such as in auto-encoder or ...
3. **Transposed Convolutions**
   - Even though the transpose of the original matrix is shown in this example, the actual values of the weights are different from the original matrix ...
   - The goal of the transposed convolution here is simply to get back the original dimensionality. It is not necessarily to get back the original feature ...
   - The elements in the downsampled vector only affect those elements in the upsampled vector that they were originally "derived" from. For example, was ...
4. **Transposed Convolutions -- drawback**
   - Transposed convolutions lead to checkerboard-style artifacts in resulting images
   - Explanation: transposed convolution yields an overlap in some feature map values
   - This leads to higher magnitude for some feature map elements than for others, resulting in the checkerboard pattern
5. **(untitled)**

### Separable Convolutions and Flattening
`cnn2/slides-separable-convolutions-flattening.tex` - 7 frames

1. **Separable convolutions**
   - Separable Convolutions are used in some neural net architectures, such as the MobileNet
   - Motivation: make convolution computationally more efficient
   - One can perform
2. **Spatially separable convolution**
3. **Depthwise separable convolution**
   - The depthwise separable convolutions, which is much more commonly used in deep learning (e.g. in MobileNet and Xception)
   - This convolution separates convolutional process into two stages of depthwise and pointwise
4. **Depthwise convolution**
   - Take number of kernels equal to the number of input channels, each kernel having depth 1. Example, if we have a kernel of size and an input of size ...
   - Every channel thus has 1 kernel associated with it. This kernel is convolved over the associated channel separately resulting in 16 feature maps
   - Stack all these feature maps to get the output volume with output size and 16 channels
5. **Pointwise convolution**
   - Take a conv with number of filters equal to number of channels you want as output
   - Perform basic convolution applied in conv to the output of the Depth-wise convolution
6. **Flattening**
7. **(untitled)**


## cnn3/  -  Modern CNN architectures

### Modern Architectures - I
`cnn3/slides-modern-cnn-1.tex` - 8 frames

1. **LeNet Architecture**
   - Pioneering work on CNNs by Yann Lecun in 1998
   - Applied on the MNIST dataset for automated handwritten digit recognition
   - Consists of convolutional, "subsampling" and dense layers
2. **AlexNet**
   - AlexNet, which employed an 8-layer CNN, won the ImageNet Large Scale Visual Recognition (LSVR) Challenge 2012 by a phenomenally large margin
   - The network trained in parallel on two small GPUs, using two streams of convolutions which are partly interconnected
   - The architectures of AlexNet and LeNet are very similar, but there are also significant differences
3. **VGG Blocks**
   - The block composed of convolutions with kernels with padding of 1 (keeping height and width) and max pooling with stride of 2 (halving the resolution ...
   - The use of blocks leads to very compact representations of the network definition
   - It allows for efficient design of complex networks
4. **VGG Network**
   - Architecture introduced by Simonyan and Zisserman, 2014 as Very Deep Convolutional Network
   - A deeper variant of the AlexNet
   - Basic idea is to have small filters and Deeper networks
5. **NiN Blocks**
   - The idea behind NiN is to apply a fully-connected layer at each pixel location (for each height and width). If we tie the weights across each spatial ...
   - The NiN block consists of one convolutional layer followed by two convolutional layers that act as per-pixel fully-connected layers with ReLU ...
   - The convolution window shape of the first layer is typically set by the user. The subsequent window shapes are fixed to
6. **Global average pooling**
   - Problem setting: tackle overfitting in the final fully connected layer
   - Classic pooling removes spatial information and is mainly used for dimension and parameter reduction
   - The elements of the final feature maps are connected to the output layer via a dense layer. This could require a huge number of weights increasing ...
7. **Network in Network (NiN)**
   - NiN uses blocks consisting of a convolutional layer and multiple convolutional layers. This can be used within the convolutional stack to allow for ...
   - NiN removes the fully-connected layers and replaces them with global average pooling (i.e., summing over all locations) after reducing the number of ...
   - Removing the fully-connected layers reduces overfitting. NiN has dramatically fewer parameters
8. **(untitled)**

### Modern Architectures - II
`cnn3/slides-modern-cnn-2.tex` - 7 frames

1. **Inception modules**
   - The Inception block is equivalent to a subnetwork with four paths
   - It extracts information in parallel through convolutional layers of different window shapes and max-pooling layers
   - convolutions reduce channel dimensionality on a per-pixel level. Max-pooling reduces the resolution
2. **GoogLeNet Architecture**
   - GoogLeNet connects multiple well-designed Inception blocks with other layers in series
   - The ratio of the number of channels assigned in the Inception block is obtained through a large number of experiments on the ImageNet dataset
   - GoogLeNet, as well as its succeeding versions, was one of the most efficient models on ImageNet, providing similar test accuracy with lower ...
3. **Residual Block (Skip connections)**
   - But: this skipping would imply learning an identity mapping. It is very hard for a neural net to learn such a 1:1 mapping through the many non-linear ...
   - Solution: offer the model explicitly the opportunity to skip certain layers if they are not useful
   - Introduced in He et. al, 2015 and motivated by the observation that stacking evermore layers increases the test- as well as the train-error ( ...
4. **ResNet Architecture**
   - The residual mapping can learn the identity function more easily, such as pushing parameters in the weight layer to zero
   - We can train an effective deep neural network by having residual blocks
   - Inputs can forward propagate faster through the residual connections across layers
5. **From ResNet to DenseNet**
   - ResNet significantly changed the view of how to parametrize the functions in deep networks
   - DenseNet (dense convolutional network) is to some extent the logical extension of this [Huang et al., 2017]
   - Dense blocks where each layer is connected to every other layer in feedforward fashion
6. **U-Net**
   - U-Net is a fully convolutional net that makes use of upsampling (via transposed convolutions, for example) as well as skip connections
   - Input images are getting convolved and down-sampled in the first half of the architecture
   - Then, they are getting upsampled and convolved again in the second half to get back to the input dimension
7. **(untitled)**


## rnn/  -  Recurrent nets, LSTM/GRU, attention

### Recurrent Neural Networks - Introduction
`rnn/slides-introduction.tex` - 6 frames

1. **Motivation for Recurrent Networks**
   - The two types of neural network architectures that we've seen so far are fully-connected networks and CNNs
   - Their input layers have a fixed size and (typically) only handle fixed-length inputs
   - The primary reason: if we vary the size of the input layer, we would also have to vary the number of learnable weights in the network
2. **RNNs - Introduction**
   - Suppose we have some text data and our task is to analyse the sentiment in the text
   - For example, given an input sentence, such as "This is good news.", the network has to classify it as either 'positive' or 'negative'
   - We would like to train a simple neural network (such as the one below) to perform the task
3. **Application example - Sentiment Analysis**
   - At, we feed the word "This" to the network and obtain
   - At, we feed the second word to the network to obtain
   - At, we feed the next word in the sentence
4. **Parameter Sharing**
   - This way, the network can process the sentence one word at a time and the length of the network can vary based on the length of the sequence
   - It is important to note that no matter how long the input sequence is, the matrices and are the same in every time-step. This is another example of ...
   - Therefore, the number of weights in the network is independent of the length of the input sequence
5. **RNNs - Use Case specific architectures**
   - Sequence-to-One: Sentiment analysis, document classification
   - One-to-Sequence: Image captioning
   - Sequence-to-Sequence: Language modelling, machine translation, time-series prediction
6. **Bidirectional RNNs**
   - A generalization of the simple RNN are bidirectional RNNs
   - These allow us to process sequential data depending on both past and future inputs, e.g. an application predicting missing words, which probably ...
   - One RNN processes inputs in the forward direction from to computing a sequence of hidden states, another RNN in the backward direction from to ...

### Recurrent Neural Networks - Backpropogation
`rnn/slides-backprop.tex` - 2 frames

1. **Simple Example: Character Level Language Model**
   - Suppose we only had a vocabulary of four possible letters: h, e, l and o
   - We want to train an RNN on the training sequence hello
   - This training sequence is in fact a source of 4 separate training examples
2. **Long-Term Dependencies**
   - It follows that
   - In general, for an arbitrary time-step in the past, will contain the term (this follows from the chain rule)
   - Based on the largest eigenvalue of, the presence of the term can either result in vanishing or exploding gradients

### Modern Recurrent Neural Networks
`rnn/slides-modernrnn.tex` - 2 frames

1. **Gated Recurrent Units (GRU)**
   - For a given time step, the hidden state of the last time step is. The update gate is computed as follows
   - We use a sigmoid to transform input values to
   - Similarly, the reset gate is computed as follows
2. **GRU vs LSTM**

### Applications of RNNs
`rnn/slides-applications.tex` - 6 frames

1. **(untitled)**
2. **RNNS - Language Modelling**
   - In an earlier example, we built a 'sequence-to-one' RNN model to perform 'sentiment analysis'
   - Another common task in Natural Language Processing (NLP) is 'language modelling'
   - Input: word/character, encoded as a one-hot vector
3. **Word embeddings**
   - Instead of one-hot representations of words it is standard practice to encode each word as a dense (as opposed to sparse) vector of fixed size that ...
   - Similar words are embedded close to each other in a lower-dimensional embedding space
   - The dimensionality of these embeddings is typically much smaller than the number of words in the dictionary
4. **(untitled)**
5. **Encoder-Decoder Network**
   - For many interesting applications such as question answering, dialogue systems, or machine translation, the network needs to map an input sequence to ...
   - This is what an encoder-decoder (also called sequence-to-sequence architecture) enables us to do!
   - An input/encoder-RNN processes the input sequence of length and computes a fixed-length context vector, usually the final hidden state or simple ...
6. **(untitled)**

### Attention and Transformers
`rnn/slides-attention.tex` - 4 frames

1. **Attention**
   - In a classical decoder-encoder RNN all information about the input sequence must be incorporated into the final hidden state, which is then passed as ...
   - With a long input sequence this fixed-sized context vector is unlikely to capture all relevant information about the past
   - Each hidden state contains mostly information from recent inputs
2. **Transformers**
   - Advanced RNNs have similar limitations as vanilla RNN networks
   - RNNs process the input data sequentially
   - Difficulties in learning long term dependency (although GRU or LSTM perform better than vanilla RNNs, they sometimes struggle to remember the context ...
3. **Transformer Components**
   - Each input (e.g., word or characters; referred to as token) in the input sequence is converted into a dense vector of fixed size
   - Embeddings capture semantic information about the tokens
   - In Transformers, input embeddings are combined with positional encodings to form the final input representations
4. **(untitled)**


## ae/  -  Unsupervised learning, autoencoders, VAE

### Unsupervised Learning
`ae/slides-unsupervised-learning.tex` - 3 frames

1. **Unsupervised Learning**
   - So far, we have described the application of neural networks to
   - In supervised learning scenarios
   - Examples are: classification, regression, object detection, semantic segmentation, image captioning, etc
2. **Unsupervised Learning - Examples**
   - 1. Clustering
   - 2. Dimensionality reduction/manifold learning
   - E.g. for visualisation in a low dimensional space
3. **(untitled)**
   - an autoencoder (a special kind of neural network) for representation learning (feature extraction, dimensionality reduction, manifold learning,...) ...
   - a generative model, i.e. a probabilistic model of the data generating distribution

### Autoencoders - Basic Principle
`ae/slides-autoencoders.tex` - 7 frames

1. **Autoencoder-task and structure**
   - Autoencoders (AEs) are
   - Task: Learn a compression of the data
   - Autoencoders consist of two parts
2. **Autoencoder (AE)- computational graph**
   - An AE has two computational steps
   - the encoder, mapping to
   - the decoder, mapping to
3. **Undercomplete Autoencoders**
   - A naive implementation of an autoencoder would simply learn the identity
   - This would not be useful
   - Therefore we have a bottleneck layer: We restrict the architecture, such that
4. **Experiment: Learn to encode MNIST**
   - Let us try to compress the MNIST data as good as possible
   - We train undercomplete AEs
5. **Increasing the Capactiy of AEs**
6. **AEs as Principal Component Analysis**
   - Consider a undercomplete autoencoder with
   - encoder function, and
   - decoder function
7. **(untitled)**

### Regularized Autoencoders
`ae/slides-autoencoders-regularized.tex` - 6 frames

1. **Overcomplete AE -- problem**
2. **(untitled)**
   - Goal: choose code dimension and capacity of encoder/decoder based on the problem
   - Regularized AEs modify the original loss function to
   - prevent the network from trivially copying the inputs
3. **Sparse Autoencoder**
   - Idea: Regularization with a sparsity constraint
   - Try to keep the number of active neurons per training input low
   - Forces the model to respond to unique statistical features of the input data
4. **Denoising autoencoders (DAE)**
   - Idea: representation should be robust to introduction of noise
   - Produce corrupted version of input, e.g. by
   - random assignment of subset of inputs to 0
5. **Experiment: Encode MNIST with a DAE**
   - We will now corrupt the MNIST data with Gaussian noise and then try to denoise it as good as possible
   - To corrupt the input, we randomly add or subtract values from a uniform distribution to each of the image entries
6. **(untitled)**
   - Goal: For very similar inputs, the learned encoding should also be very similar
   - We can train our model in order for this to be the case by requiring that the derivative of the hidden layer activations are small with respect to ...
   - In other words: The encoded state should not change much for small changes in the input

### Specific Autoencoders and Applications
`ae/slides-autoencoders-specific.tex` - 3 frames

1. **Convolutional autoencoder (ConvAE)**
   - In a ConvAE, the encoder consists of convolutional layers. The decoder, on the other hand, consists of transpose convolution layers or simple ...
2. **Real-world Applications**
   - data de-noising
   - and dimensionality reduction for the purpose of visualization
3. **(untitled)**

### Manifold learning
`ae/slides-manifold-learning.tex` - 2 frames

1. **Learning Manifolds with AEs**
   - AEs training procedures involve a compromise between two forces
   - Learning a representation of a training example such that can be approximately recovered from through a decoder
   - Satisfying an architectural constraint or regularization penalty
2. **(untitled)**

### Variational Autoencoder (VAE)
`ae/slides-vae.tex` - 1 frame

1. **(untitled)**
   - Kingma and Welling, Auto-Encoding Variational Bayes, ICLR 2014
   - Rezende, Mohamed and Wierstra, Stochastic back-propagation and variational inference in deep latent Gaussian models. ICML 2014
   - Key difference in variational autoencoders are


## genmod/  -  Generative models (overview)

### Introduction to Generative Models
`genmod/slides-introduction.tex` - 10 frames

1. **Which Face Is fake?**
2. **Deep Unsupervised Learning**
   - Representation Learning
   - Examples are: manifold learning, feature learning, etc
   - Can be done by an autoencoder
3. **Density Fitting / Learning a Generative Model**
4. **Why generative models?**
   - sampling / data generation
   - outlier detection
   - missing feature extraction
5. **Application Example: Image generation**
6. **Application Example: Neural Style Transfer**
7. **Application Example: Image Inpainting**
8. **Application Example: Semantic Labels --> Images**
9. **Application Example: Generating Images from Text**
10. **(untitled)**


## gan1/  -  Generative Adversarial Networks

### Introduction to Generative Adversarial Networks (GANs)
`gan1/slides-GAN-intro.tex` - 10 frames

1. **What is a GAN?**
   - A generative adversarial network (GAN) consists of two DNNs
   - discriminator
   - Generator transforms random noise vector into fake sample
2. **What Is a GAN?**
   - Goal of generator: fool discriminator into thinking that the synthesized samples are real
   - Goal of discriminator: recognize real samples and not being fooled by generator
   - This sets off an arms race. As the generator gets better at producing realistic samples, the discriminator is forced to get better at detecting the ...
3. **Fake currency illustration**
4. **Minimax Loss for GANs**
   - is our target, the data distribution
   - The generator is a neural network mapping a latend random vector to generated sample. Even if the generator is a determinisic function, we have ...
   - is usually a uniform distribution or an isotropic Gaussian. It is typically fixed and not adapted during training
5. **GAN training: Pseudocode**
6. **GAN training: Illustration**
   - For steps, G's parameters are frozen and one performs gradient ascent on D to increase its accuracy
   - Finally, D's parameters are frozen and one performs gradient descent on G to increase its generation performance
   - Note, that G gets to peek at D's internals (from the back-propagated errors) but D does not get to peek at G
7. **Divergence measures**
   - The goal of generative modeling is to learn
   - The differences between different generative models can be measured in terms of divergence measures
   - A divergence measure quantifies the distance between two distributions
8. **Implicit Divergence measure of GANs**
   - GANs do not explicitly minimize any divergence measure
   - However, (under some assumptions!) optimizing the minimax loss is equivalent to implicitly minimizing a divergence measure
   - That is, if the optimal discriminator is found in every iteration, the generator minimizes
9. **Optimal Discriminator**
   - The optimal discriminator returns a value greater than 0.5 if the probability to come from the data is larger than the probability to come from the ...
   - Note: The optimal solution is almost never found in practice, since the discriminator has a finite capacity and is trained on a finite amount of data
   - Therefore, the assumption needed to guarantee that the generator minimizes the JSD does usually not hold in practice
10. **(untitled)**

### Challenges for GAN Optimization
`gan1/slides-GAN-challenges.tex` - 10 frames

1. **Adversarial Training**
   - The player tries to maximize its reward (minimize its loss)
   - Use SGD (with backprob) to find the optimal parameters
   - SGD has convergence guarantees (under certain conditions)
2. **Adversarial Training -Example**
   - Consider the function, where and are both scalars
   - Player A can control and Player B can control
   - This can be rewritten as
3. **Possible behaviour \#1: Convergence**
   - The partial derivatives of the losses are
   - In adversarial training, both players perform gradient descent on their respective losses
   - In order for simultaneous gradient descent to converge to a fixed point, both gradients have to be simultaneously 0
4. **Possible behaviour \#2: Chaotic behaviour**
   - Once and have different signs, every following gradient update causes huge oscillation and the instability gets worse in time, as shown in the figure
5. **Possible behaviour \#3: Cycles**
   - A discrete example: A never-ending game of Rock-Paper-Scissors where player A chooses 'Rock' player B chooses 'Paper' A chooses 'Scissors' B chooses ...
   - Takeaway: Adversarial training is highly unpredictable. It can get stuck in cycles or become chaotic
6. **Non-stationary loss surface**
   - This is in stark contrast to (full batch) gradient descent where the loss surface is stationary no matter how many iterations of gradient descent are ...
7. **Illustration of Convergence**
8. **Illustration of Convergence: Final Step**
9. **Challenges for GAN Training**
   - Non-convergence: the model parameters oscillate, destabilize and never converge
   - Mode collapse: the generator collapses which produces limited varieties of samples
   - Diminished gradient: the discriminator gets too successful that the generator gradient vanishes and learns nothing
10. **(untitled)**

### GAN variants
`gan1/slides-GAN-variants.tex` - 9 frames

1. **Non-Saturating Loss**
   - It was discovered that a relatively strong discriminator could completely dominate the generator
   - Solution: Use a non-saturating generator loss instead
   - In contrast to the minimax loss, when the discriminator gets good at identifying fake images, the magnitude of the gradient of increases and the ...
2. **Other loss functions**
3. **Architecture-variant GANs**
4. **Conditional GANs: Motivation**
   - In an ordinary GAN, the only thing that is fed to the generator are the latent variables
   - A conditional GAN allows you to condition the generative model on additional variables
   - E.g. a generator conditioned on text input (in addition to ) can be trained to generate the image described by the text
5. **Conditional GANs: Architecture**
   - In a conditional GAN, additional information in the form of vector is fed to both the generator and the discriminator
   - can then encode all variations in that are not encoded by
   - E.g. could encode the class of a hand-written number (from 0 to 9). Then, could encode the style of the number (size, weight, rotation, etc)
6. **Conditional GANs: Example**
7. **Conditional GANs: More Examples**
8. **More Generative Models**
   - Today, we learned about one kind of (directed) generative models
   - There are other interesting generative models, e.g
   - autoregressive models
9. **(untitled)**


## adver/  -  Adversarial examples and training

### Adversarial Examples
`adver/slides-adversarials-examples.tex` - 6 frames

1. **Adversarial Robustness**
   - It is critical to examine if a trained neural net is robust and reliable
   - Adversarial robustness of a model means that a model is robust to (test time) perturbations of its inputs
   - Adversarial machine learning studies techniques which attempt to fool machine learning models through malicious input
2. **Adversarial Examples**
   - An adversarial example is an input to a model that is deliberately designed to "fool" the model into misclassifying it
   - The test error of a model is only an indicator of how well the model performs with respect to samples from the data-generating distribution
   - The performance of the same model can be drastically different on samples from a completely different distribution (on the same input space)
3. **Creation of Adversarial Examples**
   - In the examples earlier, we saw that adversarial examples can seem recognizable to humans or seem like random noise/patterns
   - In the following, given a datapoint, we want to create an adversarial example that is very similar to
   - Specifically, our goal is to find an input close to the datapoint such that a pretrained model (which accurately classifies ), ends up misclassifying
4. **Example: ResNet50**
5. **Targeted Attacks**
   - It is also possible to generate adversarial examples classified virtually as any desired class. This is known as a targeted attack
   - The only difference is that, instead of trying to just maximize the loss of the correct class, we maximize the loss of the correct class while also ...
6. **(untitled)**

### Adversarial Training Basics
`adver/slides-adversarials-training-basics.tex` - 4 frames

1. **Adversarial Training**
   - To modify a trained model so that it is more resistant to such attacks, adversarial training can be performed
   - To do so, we minimize the empirical adversarial risk which measures the worst-case empirical loss of a model, if we are able to manipulate every ...
   - To solve the optimization problem, we use SGD over. In each SGD step we repeatedly choose a minibatch of size and repeat the following until a ...
2. **Linear Models**
   - In case of linear models, the inner maximization problem can be solved exactly. We show this in the case of binary classification using linear models
   - Recall, the hypothesis space for logistic regression consists of models of the form
   - For class labels, the logistic loss is
3. **MNIST example**
   - As an example, we look at the MNIST dataset, but this time we perform logistic regression and focus only on the classification of 0s vs. 1s
   - The logistic regression classifier was trained for 10 epochs with SGD on the training set
   - This model obtained a low misclassification rate of 0.0004 on the test set
4. **(untitled)**

### Adversarial Training Advances
`adver/slides-adversarials-training-advances.tex` - 3 frames

1. **Projected Gradient Descent**
   - In contrast to logistic regression, neural networks can have a bumpier
   - As a consequence, Danskin's theorem does not longer hold and the inner optimization problem must be solved approximately
   - One approximation method is projected gradient descent (PGD)
2. **Fast Gradient Sign Method**
   - Fast Gradient Sign Method (FGSM) is a special case of PGD when = and
   - As before, the projection of an arbitrary vector onto is
   - As, the elements of are either set to or depending on the sign of the corresponding component of the gradient
3. **(untitled)**
