# CTR-DeepFM
CTR Study with DeepFM Recommender System


## 1.a 
<img width="580" alt="image" src="https://github.com/user-attachments/assets/12d46cda-4beb-4879-8c97-4e03e010a41e" />

I'd like to start with the FM formula above.

It consists of two parts:

1 - Addition

2 - Inner Products

Addition part helps capture the linear (order-1) feature interaction.
In inner products, latent (hidden) vectors also come into the play, which, in turn, helps capture the order-2 interactions. 

FM part of the DeepFM algorithm does not require features i and j both appear in the same data record, which is a great advancement in comparison to the previous approaches when it comes to capturing the order-2 feature interactions.

Factorization Machine does it by computing the inner products of the latent vectors.


## 1.b

### Arch

![image](https://github.com/user-attachments/assets/55c30e86-fa00-406c-ba57-64c5efcab494)

I'd like to start with the architecture above. Looking at the big picture, we observe the DeepFM 2 consists of 2 parts: Factorization Machine and Deep Component. 

In Deep Component, there is one challenge we need to tackle with: We're dealing with highly sparse data. It is computationally very hard to train the model with such a sparse data, therefore we compress the data to a dense embedding layer. We do this by means of Latent Vector in FM. Unlike [Zhang et al., 2016]'s approach, latent feature vectors (V) in FM now serves as network weights and they are used to do the aforementioned compression.

![image](https://github.com/user-attachments/assets/119e7910-8d39-41ea-a3b5-994dea049302)

### Math

![image](https://github.com/user-attachments/assets/ea8a1da4-64f2-4739-925b-f31d172e2a72)


Moving to the mathemathical part of DeepFM, it is trained like any neural network. In the picture above, representations are as follows: 
σ(): activation function
a(l): output
W(l): model weights 
b(l): bias
of the l-th layer.

This feed-forward NN captures high-order feature interactions, note that the low-order feature interactions are captured by the FM part.


At the end of the network we get a value, and the final output of the DeepFM model is calculated via sigmoid function, which takes the results of FM part and Deep part respectively.


## 1.c

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

Activation Function
ReLU performs better than tanh because it incudes (produces) sparsity. 

Dropout Rate
Formally, Dropout [Srivastava et al., 2014] refers to the probability that a neuron is kept in the network. In my interpretation, we kind of make some neurons sit idle in some iterations of the NN. As it can be seen from the graph below, Dropout Regularization Technique makes the model more robust when it is set to 0.9. 

![image](https://github.com/user-attachments/assets/eadab816-1f44-43ef-91f5-d73c47455a62)














 
