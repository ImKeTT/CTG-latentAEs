* [Tutorials for Latent AEs](#tutorials4latentAEs)

* [CTG via Latent AEs Survey Paper List](#ctgvialatentaes)

  - [Supervised](#supervised)
    - [2021](#2021-1)
    - [2020](#2020-1)
    - [2019](#2019-1)
    
  - [Semi-Supervised](#semi-supervised)
  
    - [2021](#2021-2)
  
    - [2020](#2020-2)
  
    - [2019](#2019-2)
    - [2018 and older](#2018-2)
  
  - [Self-Supervised](#self-supervised)
  
    - [2021](#2021-3)
    - [2020](#2020-3)
    - [2019](#2019-3)
    - [2018 and older](#2018-3)

Papers about controllable text generation (CTG) via latent auto-encoders (AEs). Include some classical style transfer generation methods but without dialogue generation for now.

<h1 id="tutorials4latentAEs">Tutorials for Latent AEs</h1>
Mostly for Variational Auto-Encoders (VAEs)

1. **Reasearchgate (2020, THU)** / [The Road from MLE to EM to VAE: A Brief Tutorial](https://www.researchgate.net/publication/342347643_The_Road_from_MLE_to_EM_to_VAE_A_Brief_Tutorial) / TL;DR
2. **EMNLP (2018, Harvard)** / [A Tutorial on Deep Latent Variable Models of Natural Language](https://arxiv.org/abs/1812.06834) / TL; DR
3. **Arxiv (2016, Carl Doersch)** / [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) / Complete and the first VAE tutorial, last updated on Jan. 2021

<h1 id="ctgvialatentaes">CTG via Latent AEs Survey Paper List</h1>
Paper list of CTG via latent AEs. I categorized all methodologies by their training paradigm (i.e., supervised, semi-supervised, self-supervised).  

- Hard Control: Knowledge/Keyword/Table-Driven controllable generation is denoted as **K2T**;
- Soft Control: Globally Sentiment / Tense / Topic controllable generation is denoted as **G2T**.

List format follows: 

 *Publication info. / paper and link / TL; DR / Code link (if available) / Chinese Blog Link (if available)* 

<h2 id="supervised">Supervised</h2>

### 2021

1. To be continued.. 

### 2020

1. 

### 2019

1. **EMNLP (Tsinghua)** / [Long and Diverse Text Generation with Planning-based Hierarchical Variational Model](https://arxiv.org/abs/1908.06605) / **K2T**, 2 latent variable models for keywords assignment plan of every sentence and word generation respectively. / [Code](https://github.com/ZhihongShao/Planning-based-Hierarchical-Variational-Model)
2. **ICASSP (Alibaba)** / [Improve Diverse Text Generation by Self Labeling Conditional Variational Auto Encoder](https://arxiv.org/abs/1903.10842) / **K2T**, 

<h2 id="semi-supervised">Semi-Supervised</h2>

### 2021

1. **Arxiv (Buffalo Univ.)** / [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation](https://arxiv.org/abs/2101.00828) / **G2T**, explored 3 different methods for condition combination with GPT-2 as both encoder and decoder of a text VAE. / Nan

### 2020

1. **ACL (Wuhan Univ.)** / [Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders](https://arxiv.org/abs/1911.03882) / **G2T**, the first "Plug-and-Play" latent AE consists of a pretrain VAE and $n$ plug-in VAE for $n$ given conditions. / [Code](https://github.com/WHUIR/PPVAE) / [Chinese Blog](https://zhuanlan.zhihu.com/p/442201826)
2. **ACL (Duke)** / [Improving Disentangled Text Representation Learning with Information-Theoretic Guidance](https://arxiv.org/abs/2006.00693) / **G2T**, explained with variation of information theory. 2 encoders for style and context encoding to produce distinct latents, a discriminator with style label for style latent adversarial learning and a VAE for context learning, concat two latents for controllable generation. / Nan
3. **EMNLP (EPFL)** / [Plug and Play Autoencoders for Conditional Text Generation]() / **G2T**, style transfer task
4. **ICLR (ByteDance)** / [Variational Template Machine For Data-to-Text Generation](https://arxiv.org/abs/2002.01127) / **K2T**, use VAE to generate keyword templates, fill pre-assigned keywords into sampled template. / [Code](https://github.com/ReneeYe/VariationalTemplateMachine)

### 2019

1. To be continued..

### 2018 and older

1. **NIPS (Michigan Univ.)** / [Content preserving text generation with attribute controls](https://arxiv.org/abs/1811.01135) / **G2T**, TL; DR

<h2 id="self-supervised">Self-Supervised</h2>

### 2021

1. **Findings (Manchester Univ.)** / [Disentangling Generative Factors in Natural Language with Discrete Variational Autoencoders](https://arxiv.org/abs/2109.07169) / **G2T**, model every condition into a discrete latent and uses Gumbel softmax for back-prop. Decomposes KL regularization loss into 3 terms related to disentanglement learning like the one described in [TC-VAE](https://arxiv.org/pdf/1802.04942.pdf)  / Nan
2. **Arxiv (EPFL)** / [Bag-of-Vectors Autoencoders For Unsupervised Conditional Text Generation]() / **G2T**, style transfer task / 

### 2020

1. **NeurIPS (UMBC)** / [A Discrete Variational Recurrent Topic Model without the Reparametrization Trick](https://arxiv.org/abs/2010.12055) / **G2T**, model word-level topic latent codes using continued multiplication approximation, and several auxiliary loss w.r.t. word-level and document-level topic correlation optimization.  / [Code](https://github.com/mmrezaee/VRTM.)
2. **ICML (MIT)** / [Educating Text Autoencoders: Latent Representation Guidance via Denoising](https://arxiv.org/abs/1905.12777) / **G2T**, add noise at input token level to avoid token-latent irrelevance issue of text latent AEs. / [Code](https://github.com/shentianxiao/text-autoencoders)
3. **ICML(ByteDance)** / [Dispersed Exponential Family Mixture VAEs for Interpretable Text Generation](https://arxiv.org/abs/1906.06719) / **G2T**, mix gaussian model (1 gaussian 1 topic ideally) for VAE prior modeling. / [Code](https://github.com/wenxianxian/demvae)
4. **ICML (Borealis)** / [On Variational Learning of Controllable Representations for Text without Supervision](https://arxiv.org/abs/1905.11975) / **G2T**, first identify the latent vacancy issue in text VAE, use GloVe and RNN embedding as two distinct latents ($z_1,z_2$). Imposes orthogonal and reconstructing regularization loss on $z_1$. / [Code](https://github.com/BorealisAI/CP-VAE) / [Chinese Blog](https://zhuanlan.zhihu.com/p/442182499)

### 2019

1. **EMNLP (CAS)** / [A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features](https://aclanthology.org/D19-1513/) / **G2T**, model text semantic and structural features via 2 separate VAEs, concat the distinct latent codes for controllable generation. / Nan
2. **NAACL (Duke)** / [Topic-Guided Variational Autoencoders for Text Generation](https://arxiv.org/abs/1903.07137) / **G2T**, consists of a latent topic model whose latent is a GMM (each Gaussian is a topic ideally) and modeled by Householder Flow, and a sequence VAE that takes the same latent for generation.  / Nan
3. **EMNLP (Buffalo Univ.)** / [Implicit Deep Latent Variable Models for Text Generation](https://arxiv.org/abs/1908.11527) / **G2T**, add an auxiliary mutual information between observed data and latent variable based on vanilla text VAE in order to educate a more meaningful latent space. / [Code](https://github.com/fangleai/Implicit-LVM)

### 2018 and older

1. **AISTATS (Duke)** / [Topic Compositional Neural Language Model](https://arxiv.org/abs/1712.09783) / **G2T**, a VAE to model topic distributions of documents and a muti-expert LSTM network for controllable generation. / Nan
2. **Arxiv (UCSB)** / [Dirichlet Variational Autoencoder for Text Modeling](https://arxiv.org/abs/1811.00135) / **G2T**, a plain VAE for sequence modeling ,and a VAE parameterized by Dirichlet for topic modeling whose latent posterior is conditioned on the sequence latent. / Nan
3. **ICML (CMU)** / [Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/abs/1702.08139) / **G2T**, self-supervised and semi-supervised