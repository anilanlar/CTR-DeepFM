# Running the Application

## Prerequisites
Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

## Installation
1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Install the required dependencies by running:

```sh
pip install -r requirements.txt
```

## Running the Application
After installing the dependencies, run the following command to start the application:

```sh
python3 DeepFM_with_csv_input.py
```

To remove components such as CrossNet or Deep part and Factorization Machine part of the DeepFM you can modify the file model_zoo/DeepFM/DeepFM_torch/src/DeepFM.py


# Introduction to CTR Prediction Problem

Click-Through Rate (CTR) prediction is a fundamental problem in online advertising, e-commerce, and recommendation systems. It involves estimating the probability that a user will click on a given item, such as an advertisement, product, or content, based on historical data and user behavior.

CTR prediction is crucial for optimizing user experience and maximizing revenue. Accurate CTR models help in delivering personalized content, allocating advertising budgets efficiently, and improving overall engagement. However, the task is challenging due to the sparsity of user interactions, high-dimensional feature spaces, and the need to capture complex patterns, such as feature interactions and contextual dependencies.

Advanced models like DeepFM, Wide & Deep, and CrossNet have been developed to address these challenges by leveraging both linear and non-linear feature interactions, combining traditional machine learning techniques with deep learning for enhanced performance.


# Part 1: Literature Survey

CTR Study with DeepFM Recommender System


## 1.1.a 
<img width="580" alt="image" src="https://github.com/user-attachments/assets/12d46cda-4beb-4879-8c97-4e03e010a41e" />

I'd like to start with the FM formula above.

It consists of two parts:

1 - Addition

2 - Inner Products

Addition part helps capture the linear (order-1) feature interaction.
In inner products, latent (hidden) vectors also come into the play, which, in turn, helps capture the order-2 interactions. 

FM part of the DeepFM algorithm does not require features i and j both appear in the same data record, which is a great advancement in comparison to the previous approaches when it comes to capturing the order-2 feature interactions.

Factorization Machine does it by computing the inner products of the latent vectors.


## 1.1.b

### Architecture

![image](https://github.com/user-attachments/assets/55c30e86-fa00-406c-ba57-64c5efcab494)


I'd like to start with the architecture above. Looking at the big picture, we observe the DeepFM 2 consists of 2 parts: Factorization Machine and Deep Component. 

In Deep Component, there is one challenge we need to tackle with: We're dealing with highly sparse data. It is computationally very hard to train the model with such a sparse data, therefore we compress the data to a dense embedding layer. We do this by means of Latent Vector in FM. Unlike [Zhang et al., 2016]'s approach, latent feature vectors (V) in FM now serves as network weights and they are used to do the aforementioned compression.

![image](https://github.com/user-attachments/assets/119e7910-8d39-41ea-a3b5-994dea049302)


### Mathematic

![image](https://github.com/user-attachments/assets/ea8a1da4-64f2-4739-925b-f31d172e2a72)



Moving to the mathemathical part of DeepFM, it is trained like any neural network. In the picture above, representations are as follows: 
σ(): activation function
a(l): output
W(l): model weights 
b(l): bias
of the l-th layer.

This feed-forward NN captures high-order feature interactions, note that the low-order feature interactions are captured by the FM part.


At the end of the network we get a value, and the final output of the DeepFM model is calculated via sigmoid function, which takes the results of FM part and Deep part respectively.


## 1.1.c

### Performance Evaluation

There are a number of key takeaways here:

1- Learning feature interaction improves the performance of CTR prediction models. We can conclude it from the fact that Logistic Regression algorithm (which does not having feature interaction) performs worse than the models having feature interaction.

2- DeepFM (it learns the low-order feature interactions and high-order feature interactions simultaneously) outperforms models that learn only low-order feature interactions and models that learn only high-order feature interactions FNN, IPNN, OPNN, PNN∗

3- Some models like LR & DNN and FM & DNN use separate feature embeddings while learning high-order and low-order feature interactions. Using the same feature embedding like we do in DeepFM outperforms the aforementioned models. 

- Do create feature interaction in recommendation systems.
- Use the same feature embedding for low-order and high-order feature interactions.
- Learn low-order and high-order feature interactions simultaneously.

Below, see the performance comparisons with Area Under Curve and Logloss methods:
![image](https://github.com/user-attachments/assets/af37bf80-a6ea-4065-b55d-0ea75338f62e)

We can see from the graph below that training of DeepFM is computationally fast:
![image](https://github.com/user-attachments/assets/e1a21d86-7f6c-4e27-99d4-6fd7e786b8c5)


### Hyper-Parameter Study

 1) activation functions; 2) dropout rate; 3) number of neurons per
layer; 4) number of hidden layers; 5) network shape

#### Activation Function
ReLU performs better than tanh because it incudes (produces) sparsity. 

Dropout Rate
Formally, Dropout [Srivastava et al., 2014] refers to the probability that a neuron is kept in the network. In my interpretation, we kind of make some neurons sit idle in some iterations of the NN. As it can be seen from the graph below, Dropout Regularization Technique makes the model more robust when it is set to 0.9. 

![image](https://github.com/user-attachments/assets/eadab816-1f44-43ef-91f5-d73c47455a62)


#### Number of Neurons per Layer
More neurons may seem to give a better performance in the NN, however, when the number of neurons per layer is increased from 400 to 800 in DeepFM it performs worse. The reason is that it creates complexity and likelihood of overfitting. See the picture below:

![image](https://github.com/user-attachments/assets/1222f7f2-1269-4ff3-8290-26ab59c2d731)


#### Number of Hidden Layers
Adding more hidden layers initially improves model performance, but excessive hidden layers lead to performance degradation due to overfitting. See the picture below

![image](https://github.com/user-attachments/assets/cfa51689-b5a9-42a9-89a8-6e009ea6ae53)

#### Network Shape
Among the network shapes such as constant, increasing, decreasing, and diamond; constant performs the best empirically. See the picture below:

![image](https://github.com/user-attachments/assets/ce26e8a1-d5a4-4cc8-ab89-5635e1db2cdd)

## 1.1.d
## Architectural Changes:

- Explicit Feature Interactions: 
  - xDeepFM introduces the Compressed Interaction Network (CIN), which directly models higher-order feature interactions at the vector level, whereas DeepFM relies on a DNN to implicitly capture these interactions.

- Hybrid Approach: 
  - xDeepFM combines both CIN and DNN, allowing it to learn both explicit and implicit feature interactions, giving it a more complete ability to capture complex patterns in the data.

## Mathematical Improvements:

- CIN’s Mechanism: 
  - CIN calculates higher-order interactions through Hadamard products and weighted sums. With each layer, the model captures increasingly complex interactions, making them easier to understand and interpret.

- Vector-Wise Operations: 
  - While DeepFM models interactions at the bit level, CIN works at the vector level, which aligns more with how factorization machines traditionally work.

- Parameter Efficiency: 
  - CIN reduces complexity by using a compressed representation, making it more scalable. It also uses methods like rank decomposition for better dimensionality management.



## 1.2. IMPROVEMENT



# Part 2: Experimental Study / Coding

### Validation and Test Metrics for DeepFM Model and Ablations

### Without Ablating Any Component:
![image](https://github.com/user-attachments/assets/fd25bc72-355b-4190-9787-f4bcc6555915)

- **Validation**:  
  - **Log Loss**: 0.277496 
  - **AUC (Area Under Curve)**: 0.941540
- **Test**:  
  - **Log Loss**: 0.280279
  - **AUC (Area Under Curve)**: 0.940040


### Ablating `fm_layer`:
![image](https://github.com/user-attachments/assets/c691ac49-95c0-4b2f-92f7-c353512ef7be)

- **Validation**:  
  - **Log Loss**: 0.279265
  - **AUC (Area Under Curve)**: 0.941655 
- **Test**:  
  - **Log Loss**: 0.281922  
  - **AUC (Area Under Curve)**: 0.940112

### Ablating `lr_layer`:
![image](https://github.com/user-attachments/assets/cb2f4e8b-df65-4900-b4b3-0718962b112f)

- **Validation**:  
  - **Log Loss**: 0.279698
  - **AUC (Area Under Curve)**: 0.940230
- **Test**:  
  - **Log Loss**: 0.281219
  - **AUC (Area Under Curve)**: 0.939263

### Ablating `MLP`:
![image](https://github.com/user-attachments/assets/2a0e1605-4760-4533-8dfe-b617219b602e)

![image](https://github.com/user-attachments/assets/815f3660-819e-4ad6-8827-7afe50b2dbe2)


- **Validation**:  
  - **Log Loss**: 0.294823  
  - **AUC (Area Under Curve)**: 0.935569  
- **Test**:  
  - **Log Loss**: 0.296791  
  - **AUC (Area Under Curve)**: 0.933542

---

### Merging CrossNet by Concatenating to the MLP Layer

![image](https://github.com/user-attachments/assets/4a0251d9-ff17-4121-a975-eb71f5520d71)

![image](https://github.com/user-attachments/assets/27bc598b-9923-4963-93f6-05cecf6f09f5)

#### Metrics:
- **Validation**:  
  - **Log Loss**: 0.278291  
  - **AUC (Area Under Curve)**: 0.940934  
- **Test**:  
  - **Log Loss**: 0.279972  
  - **AUC (Area Under Curve)**: 0.940006  


### Discussion
According to the Log Loss and AUC results, we can claim that there exists high-order feature interaction since the ablation of the deep part of the DeepFM resulted in a higher loss in AUC and higher increase in Log Loss.

## References

1. Wang, R., Fu, B., Fu, G., & Wang, M. (2018). xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. [Paper Link](https://arxiv.org/pdf/1803.05170)

2. Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. [Paper Link](https://arxiv.org/pdf/1703.04247)

3. RecZoo: Benchmarking Advances in Recommendation Systems. GitHub Repository: [BARS](https://github.com/reczoo/BARS/tree/main?tab=readme-ov-file)

4. RecZoo: FuxiCTR - A Flexible CTR Prediction Framework. GitHub Repository: [FuxiCTR v2.0.2](https://github.com/reczoo/FuxiCTR/tree/v2.0.2?tab=readme-ov-file)

