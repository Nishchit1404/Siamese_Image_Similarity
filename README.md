# Image Similarity Recognition: Siamese Networks with Binary Cross-Entropy and Triplet Loss

## 🚀 Project Overview

This repository demonstrates robust image similarity recognition using **Siamese Networks**, addressing two distinct but related problem statements. It showcases two powerful approaches: one leveraging **Binary Cross-Entropy Loss** for direct pairwise comparisons, and another employing **Triplet Loss** for learning a highly discriminative embedding space. Both solutions utilize **EfficientNet-B0** as the backbone feature extractor, highlighting expertise in leveraging pre-trained models for computer vision tasks.

The project not only provides working implementations but also delves into the nuances of dataset preparation for each method and presents detailed performance metrics. This work underscores strong foundational knowledge in deep learning architectures, loss functions, and evaluation methodologies for similarity-based tasks.

## ✨ Problem Statements & Solutions

### 1. Pairwise Image Similarity (Binary Classification)

**Problem:** Design a model that takes two images and predicts whether they belong to the same class (binary output: `1` for same, `0` for different).

**Solution: Siamese Network with Binary Cross-Entropy Loss**

- **Architecture:** A classic Siamese Network, comprising two identical "twin" subnetworks that share weights. Each subnetwork processes one image to extract features.
    
- **Loss Function:** **Binary Cross-Entropy Loss** is applied to the output of a final layer (e.g., a sigmoid activation) that combines the features, directly optimizing for binary classification of similarity.
    
- **Diagram:** ![](https://i.imgur.com/8zNrSFw.png)
    
- **Expertise Showcase:** This approach demonstrates proficiency in:
    
    - Implementing and configuring standard Siamese Networks.
        
    - Applying binary classification techniques to similarity problems.
        
    - Understanding and utilizing **Binary Cross-Entropy Loss** for appropriate tasks.
        
    - Designing models for efficient pairwise comparisons.
        

### 2. Triplet-based Image Similarity (Binary Classification)

**Problem:** Design a model that takes three images and predicts whether at least two of them belong to the same class (binary output).

**Solution: Modified Siamese Network with Triplet Loss**

- **Architecture:** An extension of the Siamese Network, featuring three identical subnetworks. The core idea is to map input images into a latent (embedding) space where similar images are close together and dissimilar images are far apart.
    
- **Loss Function:** **Triplet Loss** is employed. This loss function requires three inputs: an **Anchor** image, a **Positive** image (same class as Anchor), and a **Negative** image (different class from Anchor). The objective is to minimize the distance between the Anchor and Positive embeddings while maximizing the distance between the Anchor and Negative embeddings, ensuring a specified margin.
    
- **Diagram:** ![](https://i.imgur.com/dcOyejC.png)
    
- **Expertise Showcase:** This method highlights advanced skills in:
    
    - Designing and training models for learning robust embedding spaces.
        
    - Implementing and effectively using **Triplet Loss**, a crucial technique for tasks like face recognition, few-shot learning, and content-based image retrieval.
        
    - Understanding the complexities of sample selection (Anchor, Positive, Negative) for metric learning.
        
    - Applying distance-based classification in a learned latent space, including dynamic thresholding.
        

## 🧠 Core Architecture: Feature Extractor

For both problem statements, **EfficientNet-B0** is utilized as the feature extractor within the Siamese Network framework. EfficientNet-B0 is a highly efficient and accurate convolutional neural network, showcasing the ability to leverage state-of-the-art pre-trained models for feature extraction, reducing training time and improving performance on complex image datasets.

## 📊 Dataset: Imagenette

The project leverages the **Imagenette dataset**, a smaller, faster-to-train subset of 10 easily classified classes from the larger [Imagenet dataset](https://github.com/fastai/imagenette "null").

### Dataset Creation Strategy

A meticulous approach was taken for dataset preparation to ensure balanced and representative data for both problem statements:

- **Balanced Sampling:** Both training and testing sets maintain a balanced distribution of positive and negative pairs (for PS1) or carefully constructed triplets (for PS2).
    
- **Train/Test Split:** An 80:20 train:test split is maintained across all tasks (1280 training datapoints, 320 test datapoints).
    
- **Controlled Generation:** Custom scripts ensure random sampling while maintaining the required balance and structure for each problem's specific input format.
    

## 📈 Results

The models demonstrate strong performance across both tasks, validating the effectiveness of the chosen architectures and loss functions.

### Results for Problem Statement 1 (Siamese Network with Binary Cross-Entropy Loss):

**_Visual Representation of Convergence:**_
![](https://i.imgur.com/fGT6dkO.png)

**_Classification Report_ :**
<img width="442" alt="Screenshot 2023-09-04 at 6 39 17 PM" src="https://github.com/Nishchit1404/Siamese_Image_Similarity/assets/51109601/5f7541b3-2ce3-4b3d-a227-d62063876c36">

### Results for Problem Statement 2 (Modified Siamese Network with Triplet Loss):

The Triplet Loss model focuses on creating an embedding space where distances directly indicate similarity. Prediction is achieved by comparing distances between embeddings against a dynamically determined **threshold**.

- **Learning Latent Space:** The model effectively learns a latent space that clusters similar images and separates dissimilar ones.
    
- **Validation through Embeddings:** Validation involves verifying the learned distribution of embeddings by comparing distances of Anchor, Positive, and Negative sets.
    
- **Threshold-based Prediction:** The final classification relies on applying a threshold to the calculated distances in the embedding space. The optimal threshold depends on the dataset characteristics and the desired balance between precision and recall.
    

**_Visual Representation of Triplet Loss Convergence:**_
![](https://i.imgur.com/lNqFJ0I.png)

**_Classification Report for Threshold_ = 6:**
![](https://i.imgur.com/bjcXZsL.png)
## 💡 Key Learnings & Future Enhancements

This project demonstrates a comprehensive understanding of image similarity tasks using advanced deep learning techniques. Key takeaways include:

- **Understanding Loss Functions:** Proficient application of both **Binary Cross-Entropy** for direct classification and **Triplet Loss** for metric learning and embedding space optimization.
    
- **Architectural Flexibility:** Adapting Siamese Networks for different input configurations (pairs vs. triplets).
    
- **Efficient Feature Extraction:** Effective utilization of **EfficientNet-B0** for robust image feature representation.
    
- **Data Preparation Importance:** The critical role of meticulously prepared datasets for training similarity models.
    
- **Metric Learning Evaluation:** Understanding how to evaluate models that learn embedding spaces, including the use of thresholds for classification.
    

**Potential Future Enhancements:**

- Explore other loss functions for metric learning (e.g., Contrastive Loss, Quadruplet Loss).
    
- Implement hard negative mining strategies for Triplet Loss to further improve embedding quality.
    
- Investigate different backbone architectures and their impact on performance.
    
- Deploy the models as a web service for real-time image similarity predictions.
    
- Evaluate performance on larger and more diverse datasets.
