* [Tutorials for Latent AEs](#tutorials4latentAEs)

* [CTG via Latent AEs Survey Paper List](#ctgvialatentaes)

  - [Supervised](#supervised)
    
    - [2022](#2022-1)
    
    - [2021](#2021-1)
    - [2020](#2020-1)
    - [2019](#2019-1)
    
  - [Semi-Supervised](#semi-supervised)
  
    - [2022](#2022-2)
  
    - [2021](#2021-2)
  
    - [2020](#2020-2)
  
    - [2019](#2019-2)
    - [2018 and older](#2018-2)
  
  - [Self-Supervised](#self-supervised)
  
    - [2022](#2022-3)
    
    - [2021](#2021-3)
    - [2020](#2020-3)
    - [2019](#2019-3)
    - [2018 and older](#2018-3)

Papers about controllable text generation (CTG) via latent auto-encoders (AEs). Mainly focus on open-domain sentence generation with some style transfer generation methods (without dialogue generation for now).

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
### 2022

1. **ACL Findings (Huawei)** / [Controlled Text Generation Using Dictionary Prior in Variational Autoencoders](https://aclanthology.org/2022.findings-acl.10.pdf) / **G2T**, it proposes a discrete latent prior weighted by continuous Dirichelet distribution, [iVAE](https://arxiv.org/abs/1908.11527) KL loss for training. And develops contrastive learning loss for controllable generation, and it used both LSTM and GPT-2 models as encoder&decoder with SoTA language modeling performance. / Nan

### 2021

1. **NeurIPS (UCSD)** / [A Causal Lens for Controllable Text Generation](https://arxiv.org/pdf/2201.09119.pdf) / **G2T**, the first unified causal framework for text generation under control, introduced Structured Causal Model (SCM) for conditional generation, used counterfactual and intervention causal tools for style transfer and controlled generation tasks respectively. / Nan

### 2020

1. TBD

### 2019

1. **EMNLP (Tsinghua)** / [Long and Diverse Text Generation with Planning-based Hierarchical Variational Model](https://arxiv.org/abs/1908.06605) / **K2T**, 2 latent variable models for keywords assignment plan of every sentence and word generation respectively. / [Code](https://github.com/ZhihongShao/Planning-based-Hierarchical-Variational-Model)
2. **ICASSP (Alibaba)** / [Improve Diverse Text Generation by Self Labeling Conditional Variational Auto Encoder](https://arxiv.org/abs/1903.10842) / **K2T**, 
3. **NeurIPS (PKU)** / [Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation](https://arxiv.org/abs/1905.12926) / **G2T**, style transfer generation
4. **Arxiv (Waterloo Univ.)** / [Stylized Text Generation Using Wasserstein Autoencoders with a Mixture of Gaussian Prior](https://arxiv.org/abs/1911.03828), [Corresponding Thesis Paper](https://uwspace.uwaterloo.ca/bitstream/handle/10012/16757/Ghabussi_Amirpasha.pdf;jsessionid=2418986708437ABDE6DE37FA3DB4A109?sequence=1) / **G2T**, 

<h2 id="semi-supervised">Semi-Supervised</h2>
### 2022

1. **ICML (Monash)** / [Variational Autoencoder with Disentanglement Priors for Low-Resource Task-Specific Natural Language Generation](https://arxiv.org/abs/2202.13363) / **G2T**, BERT encoder for overall feature extraction and two different MLP encoder for label and content encoding severally. Used prefix-tuning and GPT-2 decoder for zero/few-shot style transfer generation. / Nan
2. **Arxiv (Stanford)** / [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217) / **K2T**, syntactic control over continuous difussion language model in continuous word embedding space (as the latent space and optimized in VAE paradigm) with Plug and Play component. / [Code](https://github.com/XiangLi1999/Diffusion-LM)
3. **ICML (UCLA)** / [Latent Diffusion Energy-Based Model for Interpretable Text Modeling](https://arxiv.org/pdf/2206.05895.pdf) / **G2T**,  use diffusion process on latent space with prior sampling with EBM, variational bayes for latent posterior approximation. Similar paradigm of [S-VAE](https://arxiv.org/pdf/1406.5298.pdf) to deal with labels in semi-supervision. / [Code](https://github.com/yuPeiyu98/LDEBM)
4. **KBS (Tsinghua)** / [PCAE: A Framework of Plug-in Conditional Auto-Encoder for Controllable Text Generation](https://www.sciencedirect.com/science/article/pii/S0950705122008942) / **G2T**, invent *Broadcasting Net* to repeatly add control signals into latent space to create a concentrate and manipulable latent space in VAE. Experimenced on both RNN and BART VAE models. / [Code](https://github.com/ImKeTT/pcae)
5. **Arxiv (CUHK)** / [Composable Text Controls in Latent Space with ODEs](https://arxiv.org/abs/2208.00638) / **G2T**, employs diffusion process in the latent space based on adaptive GPT-2 VAE (similar to [AdaVAE](https://arxiv.org/abs/2205.05862)), the diffusion process transfer latent distribution from Gaussian to controlled one. Few parameters and data are used for training. / [Code](https://github.com/guangyliu/LatentOps)

### 2021

1. **Entropy (Wuhan Univ.)** / [A Transformer-Based Hierarchical Variational AutoEncoder Combined Hidden Markov Model for Long Text Generation](https://pdfs.semanticscholar.org/6f91/070dbffffc841cf8734872dbbd96ba8b1bfd.pdf) / **G2T**, long controllable text (passage) generation, use word-level and sentence-level latent variables. Encode the passage title as the latent prior to conduct controllable passage generation. / Nan

2. **Arxiv (EPFL)** / [Bag-of-Vectors Autoencoders For Unsupervised Conditional Text Generation]() / **G2T**, style transfer task / Nan
3. **EACL (Waterloo Univ.)** / [Polarized-VAE: Proximity Based Disentangled Representation Learning for Text Generation](https://arxiv.org/abs/2004.10809) / **G2T**, style transfer task; proposed to use two separate encoders to encode sentence syntax and semantic information, added a proximity loss (cosine) on latent space to distinguish dissimilar sentences (with different labels) / [Code](https://github.com/vikigenius/prox_vae)
4. **Arxiv (Buffalo Univ.)** / [Transformer-based Conditional Variational Autoencoder for Controllable Story Generation](https://arxiv.org/abs/2101.00828) / **G2T**, explored 3 different methods for condition combination with GPT-2 as both encoder (w/o causal mask) and decoder of a text VAE. / [Code](https://github.com/fangleai/TransformerCVAE) / [Chinese Blog](https://zhuanlan.zhihu.com/p/446370783)
5. **Arxiv (UCLA)** / [Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification](https://arxiv.org/pdf/2108.11556.pdf) / **G2T**, use energy-based model to model latent prior and variational bayes for posterior approximation,  use the similar paradigm of [S-VAE](https://arxiv.org/pdf/1406.5298.pdf) to deal with semi-supervised latent learning. / [Code](https://github.com/bpucla/ibebm)

### 2020

1. **ACL (Wuhan Univ.)** / [Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders](https://arxiv.org/abs/1911.03882) / **G2T**, the first "Plug-and-Play" latent AE consists of a pretrain VAE and $n$ plug-in VAE for $n$ given conditions. / [Code](https://github.com/WHUIR/PPVAE) / [Chinese Blog](https://zhuanlan.zhihu.com/p/442201826)
2. **ACL (Duke)** / [Improving Disentangled Text Representation Learning with Information-Theoretic Guidance](https://arxiv.org/abs/2006.00693) / **G2T**, explained with variation of information theory. 2 encoders for style and context encoding to produce distinct latents, a discriminator with style label for style latent adversarial learning and a VAE for context learning, concat two latents for controllable generation. / Nan
3. **EMNLP (EPFL)** / [Plug and Play Autoencoders for Conditional Text Generation](https://arxiv.org/abs/2010.02983) / **G2T**, style transfer task, proposed an 'offset' net to encode 
4. **ICLR (ByteDance)** / [Variational Template Machine For Data-to-Text Generation](https://arxiv.org/abs/2002.01127) / **K2T**, use VAE to generate keyword templates, fill pre-assigned keywords into sampled template. / [Code](https://github.com/ReneeYe/VariationalTemplateMachine)
5. **EMNLP (Microsoft)** / [Optimus: Organizing Sentences via Pre-trained Modeling of a Latent Space](http://arxiv.org/abs/2004.04092) / **G2T**, the FIRST VAE with big pre-trained models (BERT and GPT-2), one of its downstream tasks is bi-class controlled text generation. / [Code](https://github.com/ChunyuanLI/Optimus)

### 2019

1. TBD

### 2018 and older

1. **NIPS (Michigan Univ.)** / [Content preserving text generation with attribute controls](https://arxiv.org/abs/1811.01135) / **G2T**, style transfer task
2. **ICML (CMU)** / [Improved Variational Autoencoders for Text Modeling using Dilated Convolutions](https://arxiv.org/abs/1702.08139) / **G2T**, self-supervised and semi-supervised generation task.
3. **ICML (CMU)** / [Adversarially regularized autoencoders](https://arxiv.org/pdf/1706.04223.pdf) / **G2T**, two-stage training paradigm, first train a auto-encoder, than train a conditional GAN to produce the latent vectors. / [Code](https://github.com/jakezhaojb/ARAE)

<h2 id="self-supervised">Self-Supervised</h2>
### 2022

1. **Arxiv (Tsinghua)** / [AdaVAE: Exploring Adaptive GPT-2s in Variational Auto-Encoders for Language Modeling](https://arxiv.org/abs/2205.05862) / **G2T**, pre-trained GPT-2 as both encoder (w/o causal mask) decoder with adapter tuning method, proposed efficient *Latent Attention* for latent space construction. Conducted linear and arithmetic interpolation for text generation. / [Code](https://github.com/ImKeTT/adavae) / [Chinese Blog](https://zhuanlan.zhihu.com/p/513807583)

### 2021

1. **Findings (Manchester Univ.)** / [Disentangling Generative Factors in Natural Language with Discrete Variational Autoencoders](https://arxiv.org/abs/2109.07169) / **G2T**, model every condition into a discrete latent and uses Gumbel softmax for back-prop. Decomposes KL regularization loss into 3 terms related to disentanglement learning like the one described in [TC-VAE](https://arxiv.org/pdf/1802.04942.pdf)  / Nan

### 2020

1. **NeurIPS (UMBC)** / [A Discrete Variational Recurrent Topic Model without the Reparametrization Trick](https://arxiv.org/abs/2010.12055) / **G2T**, model word-level topic latent codes using continued multiplication approximation, and several auxiliary loss w.r.t. word-level and document-level topic correlation optimization.  / [Code](https://github.com/mmrezaee/VRTM.)
2. **ICML (MIT)** / [Educating Text Autoencoders: Latent Representation Guidance via Denoising](https://arxiv.org/abs/1905.12777) / **G2T**, add noise at input token level to avoid token-latent irrelevance issue of text latent AEs. / [Code](https://github.com/shentianxiao/text-autoencoders)
3. **ICML(ByteDance)** / [Dispersed Exponential Family Mixture VAEs for Interpretable Text Generation](https://arxiv.org/abs/1906.06719) / **G2T**, mix exponential family model (1exponential distribution for 1 topic ideally) for VAE prior modeling. / [Code](https://github.com/wenxianxian/demvae) / [Chinese Blog](https://zhuanlan.zhihu.com/p/442608395?)
4. **ICML (Borealis)** / [On Variational Learning of Controllable Representations for Text without Supervision](https://arxiv.org/abs/1905.11975) / **G2T**, first identify the latent vacancy issue in text VAE, use GloVe and RNN embedding as two distinct latents ($z_1,z_2$). Imposes orthogonal and reconstructing regularization loss on $z_1$. / [Code](https://github.com/BorealisAI/CP-VAE) / [Chinese Blog](https://zhuanlan.zhihu.com/p/442182499)

### 2019

1. **EMNLP (CAS)** / [A Topic Augmented Text Generation Model: Joint Learning of Semantics and Structural Features](https://aclanthology.org/D19-1513/) / **G2T**, model text semantic and structural features via 2 separate VAEs, concat the distinct latent codes for controllable generation. / [Chinese Blog](https://zhuanlan.zhihu.com/p/442608395?)
2. **NAACL (Duke)** / [Topic-Guided Variational Autoencoders for Text Generation](https://arxiv.org/abs/1903.07137) / **G2T**, consists of a latent topic model whose latent is a GMM (each Gaussian is a topic ideally) and modeled by Householder Flow, and a sequence VAE that takes the same latent for generation.  / [Chinese Blog](https://zhuanlan.zhihu.com/p/442608395?)
3. **EMNLP (Buffalo Univ.)** / [Implicit Deep Latent Variable Models for Text Generation](https://arxiv.org/abs/1908.11527) / **G2T**, add an auxiliary mutual information between observed data and latent variable based on vanilla text VAE in order to educate a more meaningful latent space. / [Code](https://github.com/fangleai/Implicit-LVM)
4. **ACL (Nanjing Univ.)** / [Generating Sentences from Disentangled Syntactic and Semantic Spaces](https://arxiv.org/abs/1907.05789) / **G2T**, 

### 2018 and older

1. **AISTATS (Duke)** / [Topic Compositional Neural Language Model](https://arxiv.org/abs/1712.09783) / **G2T**, a VAE to model topic distributions of documents and a muti-expert LSTM network for controllable generation. / Nan
2. **Arxiv (UCSB)** / [Dirichlet Variational Autoencoder for Text Modeling](https://arxiv.org/abs/1811.00135) / **G2T**, a plain VAE for sequence modeling ,and a VAE parameterized by Dirichlet for topic modeling whose latent posterior is conditioned on the sequence latent. / [Chinese Blog](https://zhuanlan.zhihu.com/p/442608395?)