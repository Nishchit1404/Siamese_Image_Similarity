# Image Similarity

The Objective of the the repository is to display image similarity recognition using Siamese Networks.

## Problem Statement :
1. Design a model that takes in two images and predicts whether they belong to the same class or not.
2. Design a model that takes in three images and predicts whether at least two of them belong to the same class or not.

### Problem Statement 1:
The first problem statment requires to design and train a deep learning model to predict whether two images belong to the same class or not. This is a binary classification problem. The model should take as input a pair of images and output a binary prediction ‘1’ if they belong to the same class and ‘0’ if they do not.

### Problem Statement 2:
Your second problem statment is similar to the firs but requires to design and train a model that takes in three images and predicts whether at least two of them belong to the same class or not.
This is a binary classification problem as well.

## Architectures Employed : 

### Problem statement 1 :
Siamese Networks are a type of neural network architecture that are specifically designed for tasks like image similarity and verification. They consist of two identical subnetworks (often called "**twin**" networks) that share the same weights and architecture. They process each image separately and extract visual features from them. The extracted features are then compared to determine similarity.

![](https://i.imgur.com/8zNrSFw.png)

#### EfficientNet-b0 is being employed as the feature extractor

### Problem statement 2 :
The Siamese Network is modified and used as neural network architecture. It consist of three identical subnetworks that share the same weights and architecture. The extracted Features of the three images are then compared using Triplet Loss. The network tries to map input images (real, complex features) to a latent space.

![](https://i.imgur.com/dcOyejC.png)

#### EfficientNet-b0 is being employed as the feature extractor

## Dataset : 
The Imagenette dataset is a subset of 10 easily classified classes from the [Imagenet dataset](https://github.com/fastai/imagenette).
The Dataset used for training is a Balanced subset of the Imagenet Dataset and has been selected in the following Manner :

### Dataset creation for Problem Statement 1
1. The train set has total of 1280 datapoints and test set has total of 320 datapoints for both tasks (train:test split of 80:20).
2. Each class has 128 datapoints in train set and 32 datapoints in test set.
3. Random Sampling was done by iterating through the Datapool of each class creating 128 datapoints per class and 1280 in total.
4. For the given task the probability of +ve image pairs (having same class) and -ve image pairs is 50%-50%.

### Dataset creation for Problem Statement 2
1. Each datapoint generation required sampling of an anchor image, a positive image of the same class and negative image of different class.
(Anchor, Positive and Negative were sampled because Triplet Loss was used for training).
2. The train set has total of 1280 datapoints and test set has total of 320 datapoints for both tasks (train:test split of 80:20).
3. Each class has 128 datapoints in train set and 32 datapoints in test set.
4. Random Sampling was done by iterating through the Datapool of each class.


