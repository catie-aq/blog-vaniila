---
title: "FACIAL RECOGNITION USING SIAMESE NEURAL NETWORKS"
tags:
  - CV
  - facial recognition
  - siamese networks
  - 2023
categories:
excerpt : "CV - Explaining Siamese neural networks for face recognition <br>- Difficuly: beginner"
header:
   overlay_color: "#1C2A4D"
author_profile: false
sidebar:
    nav: sidebar-cv-en
classes: wide
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script> 

# Introduction

<p style="text-align:justify;" >
<b>Facial recognition</b> is a technology that enables <b>automatic identification of people</b> based on characteristic information extracted from pictures of their face. This technology has evolved significantly over the last three decades (<a href="https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf">Bromley et al.</a> addressed a similar issue in 1994), particularly due to contributions from <b>artificial intelligence</b> and <b>deep learning</b> techniques.
<br><br>
<b>Neural networks</b> are now at the core of many devices and equipment used for identifying individuals. The design and integration of these networks naturally depend on the intended application and <b>available hardware resources</b>, as well as other important parameters such as the <b>availability of datasets for training</b>.
<br><br>
Facial recognition is often approached as a <b>classification problem</b> where a neural network is used to determine the most likely class of a picture with individual's face. However, this approach can be problematic in some cases because:<br>
it requires a substantial set of labeled data that can be tedious to build and update,<br>
- it requires a fairly substantial <b>set of labeled data</b> potentially tedious to build and update<br>
- the corresponding network must be <b>retrained</b> whenever new classes (i.e., new individuals to be identified) need to be added
<br><br>  
For instance, in cases where new individuals need to be recognized on the fly in a video stream, the <b>classification approach is inappropriate</b>. Instead, it is necessary to turn to solutions that require fewer material resources and computational time.
<br><br>
In these cases, implementation of <b>architectures based on similarity calculation functions</b> are preferred, to determine whether pictures of individuals to be identified match the representations of known individuals recorded in a database, which may be enriched in real-time as new faces are detected.
<br><br>
We present here a description of a solution of this type based on a <b>siamese architecture</b> that we have tested and implemented as part of the <b><a href="https://www.robocup.org/domains/3">RoboCup@Home</a></b>, an international competition in the field of service robotics, where robots must interact with human operators.
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/screen.jpg">
  <figcaption>
  Output of the algorithm
  </figcaption>
</figure>
</center>

<br><br>


# General architecture

<p style="text-align:justify;" >
Our facial recognition solution integrates a combination of tools and neural networks designed to perform the following tasks:<br>
- Detect faces of individuals in pictures<br>
- Create a 64-dimensional <i>identity vector</i> for each isolated face<br>
- Calculate the distance between the vectors associated with two distinct images<br>
- And determine whether the vector associated with one face is similar to another already identified by searching a database
<br><br>
Tools for face detection - in a picture or video stream - and extraction will be discussed later.
<br><br>
The core of the device consists of a model with an objective function that calculates a similarity score to determine if two face pictures refer to the same individual.
<br><br>
Our implemented architecture is <b>siamese</b> and involves two instances of the same <b>convolutional neural network</b>. Each one takes a picture of the face as input and provides a 64-dimensional <b>vector representation</b> of it as output.
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/overview.png">
  <figcaption>
  Overview of the architecture
  </figcaption>
</figure>
</center>

<p style="text-align:justify;" >
A convolutional neural network has been trained to provide <b>close representations</b> – in Euclidean distance - <b>for two face images of the same person</b>, and distant or very <b>distant representations for images of two different people</b>.
<br><br>
The outputs of the two network instances (identical in all points and therefore sharing the same configuration and the same weights) merge and are then used to calculate a <b>similarity score based on the distance between the vector representations of the input images</b>.
<br><br>
Each face detected in an image or video stream is then encoded by the network and <b>compared to a set of known fingerprints</b> stored in a database. The result of this comparison is returned as a scalar value (the similarity score mentioned earlier) and evaluated in the light of a predetermined threshold. If the similarity score exceeds the threshold, the fingerprints can be seen <b>as identical</b> and the individual is thus <b>identified</b>.
</p>
<br><br>
  
# Network characteristics and drive
  
<p style="text-align:justify;" >
The challenge here is to design and train the convolutional network so that <b>similar inputs are projected at relatively close locations in the performance space</b> and, conversely, different <b>inputs are projected at distant points</b>.
</p>
<br>

## Dataset used and pre-processing

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/base_de_donnees.png">
  <figcaption>
  <center>
  Source : 
  <a href="https://paperswithcode.com/dataset/vggface2-1">https://paperswithcode.com/dataset/vggface2-1</a>
  </center>
  </figcaption>
</figure>
</center>

<p style="text-align:justify;" >
The convolutional network used in this study was trained on the <a href="http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/">VGGFace2</a> dataset – by Cao et al. (2018)- which is a publicly accessible dataset containing around 3.3 million images of over 9000 individuals.
<br><br>
The images taken from this dataset with great variability in poses, age of subjects, exposures, etc., have been <b>normalized</b> so as to identify the faces and position the characteristic points of these (eyes, nose, mouth) in identical coordinates whatever the cliché considered.
<br><br>
The image normalization step is crucial for the network's performance. Face detection was performed using the <a href="https://arxiv.org/abs/1905.00641v2">RetinaFace</a> neural network developed by Deng et al. (2019), which identifies the <i>bounding box</i> of the face as well as characteristic points. The image is then <b>cropped and transformed</b> to position the characteristic points in predetermined positions.
<br><br>
The convolutional network positioned at the core of our facial recognition device was then trained from these pictures.
</p>
<br>

## Architecture

<p style="text-align:justify;" >
The architecture of our network is based on <a href="https://arxiv.org/abs/1905.11946">EfficientNet-B0</a>, developed by Tan and Le (2019). This choice is a compromise between various constraints relevant to our problem, as the algorithm will be embedded in the robot with limited graphics card capabilities. 
The number of parameters in memory is constrained, and the execution speed must be fast, as the people to be identified may move during the identification process.
<br><br>
This network offers relatively short inference times compared to deeper networks, which are more efficient but require significantly longer processing times.
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/reseau.png">
  <figcaption>
  Tan and Le EfficientNet-B0 Network Architecture (2019)
  </figcaption>
</figure>
</center>

<p style="text-align:justify;" >
Notes:<br>
- EfficientNet-B0 is the result of a field of research that holds an important place in deep learning: NAS (<i>Neural Architecture Search</i>), and which aims to automate and optimize the architectures of the networks used. It has given rise to many networks, the most popular of which are the <a href="https://arxiv.org/abs/1704.04861">MobileNets</a> by Howard et al. (2017), <a href="https://arxiv.org/abs/1905.11946">EfficientNet</a> (Tan and Le (2019)) or <a href="https://arxiv.org/abs/2201.03545">ConvNext</a> by Liu et al. (2022).<br>
- nowadays <i>transformers</i> for vision (<a href="https://arxiv.org/abs/2010.11929">ViT</a> by Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zha  et al. (2020)) are an alternative to convolutional neural networks. Examples include the <a href="https://arxiv.org/abs/2103.14030">Swin Transformer</a> by Liu, Lin, Cao, Hu et al. (2021) 
</p>
<br>


## Choice of objective function

<p style="text-align:justify;" >
Learning similarities requires the use of appropriate objective functions, including the <i><a href="https://ieeexplore.ieee.org/document/1640964">contrastive loss</a></i> by Hadsell et al. (2005) and the <i><a  href="https://arxiv.org/abs/1503.03832">triplet loss</a></i> by Schroff et al. (2015). 
<br><br>
The <b><i>contrastive loss</i></b> is defined by:
</p>
$$
L(v_1, v_2)=\frac{1}{2} (1-\alpha)d(v_1, v_2)² + \frac{1}{2} \alpha(max(0,m-d(v_1, v_2)))²
$$

where $$v_1$$ and $$v_2$$ being two vectors, α is a coefficient of 1 if the two vectors are of the same class, 0 otherwise, $$d$$ is a function of any distance, and $$m$$ is a real called the margin.
<br><br>
Intuitively, this objective function penalizes two vectors of  the same class by their distance, while two vectors of different classes are penalized only if their distance is less than $$m$$.
<br><br>
<p style="text-align:justify;" >
The function <b>triplet loss</b> involves a third vector, the anchor, in its equation: 
</p>
$$
L(a, v_1, v_2)=max(d(a,v_1)²-d(a,v_2)²+m, 0) 
$$

Here, $$a$$ denotes the anchor, $$v_1$$ is a vector of the same class as $$a$$ and $$v_2$$ is a vector of a different class of $$a$$.
<br><br>
This function simultaneously tends to bring the pair $$(a, v_1)$$ closer together and to move away the pair $$(a, v_2)$$ as shown in the following figure: 

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/triplet.png">
  <figcaption>
  Triplet loss by Schroff et al. (2015)
  </figcaption>
</figure>
</center>

<p style="text-align:justify;" >
In general, training networks using these objective functions directly is quite expensive, the convergence of this type of system being longer to achieve than, for example, on conventional classification problems.
<br><br>
In order to circumvent this difficulty, we have adopted an alternative approach consisting of a two-step network training.
</p>
<br>

## Training

<p style="text-align:justify;" >
We started by training the network on a classification task to recognize a person's picture from among 9,000 available identities, using a classical <b>entropy</b> function (<b><i>crossentropy</i></b>) as the cost function.
<br><br>
After achieving convergence on the classification task, we replaced the last classification layer with a new layer that represents the image embedding as the output.
<br><br>
The previous layers retained their weights from the previous training step. This approach is similar to transfer learning, where we aim to preserve the learned features during the classification task and reuse them to build the metric that interests us.
<br><br>
We then retrained the network with a <b><i>contrastive</i></b> or <b><i>triplet</i></b> objective function, as seen above.
<br><br>
This method enables us to quickly train a siamese network.
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/serveur.png">
  <figcaption>
  Source : <a href="https://www.catie.fr/cluster-vaniila/">https://www.catie.fr/cluster-vaniila/</a>
  </figcaption>
</figure>
</center>

<br><br>                                                                                                                    

# Implementation and integration

<p style="text-align:justify;" >
The facial recognition device was created by integrating various tools and scripts, mostly coded in Python.
<br><br>
The neural network itself is implemented using <a href="https://github.com/ pytorch/pytorch">PyTorch</a>, developed by Paszke, Gross, Chintala, Chanan et al. (2016), and specifically <a href="https://github.com/Lightning-AI/ lightning">Pytorch  Lightning</a>, developed by Falcon et al. (2019). The network was trained using the computational resources provided by CATIE's <a href="https://www.vaniila.ai/">VANIILA</a> platform.
<br><br>
This approach enabled us to complete the successive training sessions within a reasonable time frame of less than two hours. The results were more than interesting, with an F1 score of 0.92, which outperforms the commercial solutions we tested.
</p>
<br><br> 


# Conclusion

<p style="text-align:justify;" >
In this article, we have described the process of facial recognition using a siamese network with an adapted cost function. We first extracted and aligned faces, and then trained the network on a large dataset of labeled images to address a face recognition problem.
<br><br>
However, a major limitation of this approach is the need for a large number of labeled images, which can be expensive or impossible to obtain. To address this issue, self-supervised learning methods have been developed, which enable models to be trained on large amounts of unlabeled data. We will delve into the details of these self-supervised techniques in a future article.
<br><br>

Stay tuned!
</p>

  
<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/epock.jpg">
  <figcaption>
  Epock, CATIE's robot, during RoboCup 2019
  </figcaption>
</figure>
</center>


<br><br>

# References

<p style="text-align:justify;">
- <a href="https://arxiv.org/abs/2201.03545">A ConvNet for the 2020s</a> by Liu et al. (2022)<br>
- <a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a> by Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zha et al. (2020)<br>
- <a href="https://ieeexplore.ieee.org/document/1640964">Dimensionality Reduction by Learning an Invariant Mapping</a> by Hadsell et al. (2005)<br>
- <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> by Tan et Le (2019)<br>
- <a href="https://arxiv.org/abs/1503.03832">FaceNet: A Unified Embedding for Face Recognition and Clustering</a> by Schroff et al. (2015)<br>
- <a href="https://arxiv.org/abs/1704.04861">MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications</a> by Howard et al. (2017)<br>
- <a href="https://github.com/pytorch/pytorch">PyTorch</a> by Paszke, Gross, Chintala, Chanan et al. (2016)<br>
- <a href="https://github.com/Lightning-AI/lightning">Pytorch Lightning</a> by Falcon et al. (2019)<br>
- <a href="https://arxiv.org/abs/1905.00641v2">RetinaFace: Single-stage Dense Face Localisation in the Wild</a> by Deng et al. (2019)<br>
- <a href="https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf">Signature Verification using a "Siamese" Time Delay Neural Network</a> by Bromley et al. (1994)<br>
- <a href="https://arxiv.org/abs/2103.14030">Swin Transformer: Hierarchical Vision Transformer using Shifted Windows</a> by Liu, Lin, Cao, Hu et al. (2021)<br>
- <a href="https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf">VGGFace2: A dataset for recognising faces across pose and age</a> by Cao et al. (2018)
</p>

<br><br>

# Comments
<script src="https://utteranc.es/client.js"
        repo="catie-aq/blog-vaniila"
        issue-term="pathname"
        label="[Comments]"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
