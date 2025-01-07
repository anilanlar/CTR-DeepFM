# CTR-DeepFM
CTR Study with DeepFM Recommender System


### 1.a 
<img width="580" alt="image" src="https://github.com/user-attachments/assets/12d46cda-4beb-4879-8c97-4e03e010a41e" />

Let's start with the FM formula above.

It consists of two parts:

1 - Addition

2 - Inner Products

Addition part helps capture the linear (order-1) feature interaction.
In inner products, latent (hidden) vectors also come into the play, which, in turn, helps capture the order-2 interactions. 

FM part of the DeepFM algorithm does not require features i and j both appear in the same data record, which is a great advancement in comparison to the previous approaches when it comes to capturing the order-2 feature interactions.

Factorization Machine does it by computing the inner products of the latent vectors.
