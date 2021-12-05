[TOC] 

Papers about controllable text generation (CTG) via latent auto-encoders (AEs). Dialogue generation task is not included for now.

#  + Tutorials for latent AEs

Mostly for Variational Auto-Encoder (VAE)

1. Reasearchgate (2020, THU) / [The Road from MLE to EM to VAE: A Brief Tutorial][https://www.researchgate.net/publication/342347643_The_Road_from_MLE_to_EM_to_VAE_A_Brief_Tutorial] / TL;DR
2. Arxiv (2016, Carl Doersch) / [Tutorial on Variational Autoencoders][https://arxiv.org/abs/1606.05908] / Complete and the first VAE tutorial, last updated on Jan. 2021

# + CTG via Latent AEs Survey Papers

Paper list of CTG via latent AEs. I categorized all methodologies by their training paradigm (i.e., supervised, semi-supervised, self-supervised).  

- Hard Control: Knowledge/Keyword/Table-Driven controllable generation is denoted as **K2T**;
- Soft Control: Globally Sentiment / Tense / Topic controllable generation is denoted as **G2T**.

List format follows *Publication info. / paper and link (or blog link) / TL; DR*

##  Supervised

### 2021



### 2020



### 2019

1. EMNLP (THU) / [Long and Diverse Text Generation with Planning-based Hierarchical Variational Model][https://arxiv.org/abs/1908.06605] / **K2T**, 2 latent variable models for keywords assignment plan of every sentence and word generation respectively.
2. ICASSP (Alibaba) / [Improve Diverse Text Generation by Self Labeling Conditional Variational Auto Encoder][https://arxiv.org/abs/1903.10842] / **K2T**, 

## Semi-Supervised

### 2021

1. Arxiv (Buffalo Univ.) / [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation][https://arxiv.org/abs/2101.00828] / **G2T**, explored 3 different methods for condition combination with GPT-2 as both en/decoder.

### 2020

1. ACL (Wuhan Uni.) / [Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders][https://arxiv.org/abs/1911.03882] / **G2T**, first "Plug-and-Play" latent AE consists of a pretrain VAE and $n$ plug-in VAE for $n$ given conditions.
2. ACL (Duke) / [Improving Disentangled Text Representation Learning with Information-Theoretic Guidance][https://arxiv.org/abs/2006.00693] / **G2T**, 

### 2019





## Self-Supervised

### 2021

1. Findings (Manchester Uni.) / [Disentangling Generative Factors in Natural Language with Discrete Variational Autoencoders][https://arxiv.org/abs/2109.07169] / **G2T**, novel loss functions for text disentangle learning.

### 2020

1. NeurIPS () / [A Discrete Variational Recurrent Topic Model without the Reparametrization Trick][https://arxiv.org/abs/2010.12055] / **G2T**, 
2. ICML (MIT) / [Educating Text Autoencoders: Latent Representation Guidance via Denoising][https://arxiv.org/abs/1905.12777] / **G2T**, add noise at input token level to avoid token-latent irrelevance issue of text latent AEs.

### 2019

1. ACL(CAS) / [A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features](https://aclanthology.org/D19-1513/) / **G2T**, model text semantic and structural features via 2 separate VAEs, concat the distinct latent codes for controllable generation.
2. NAACL (Duke) / [Topic-Guided Variational Autoencoders for Text Generation][https://arxiv.org/abs/1903.07137] / **G2T**, consists of a latent topic model whose latent is a GMM (each Gaussian is a topic ideally) and modeled by Householder Flow, and a sequence VAE that takes the same latent for generation. 
3. EMNLP (Buffalo Uni.) / [Implicit Deep Latent Variable Models for Text Generation][https://arxiv.org/abs/1908.11527] / **G2T**, add an auxiliary mutual information loss between observed data and latent variable.

