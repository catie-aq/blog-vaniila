---
title: "FACIAL RECOGNITION USING SIAMESE NEURAL NETWORKS"
categories:
  - CV
tags:
  - Facial recognition using Siamese neural networks
excerpt : "CV - Explaining Siamese neural networks for face recognition"
header :
    overlay_image: "https://raw.githubusercontent.com/lbourdois/blog/master/assets/images/NLP_radom_blog.png"
author_profile: false
sidebar:
    nav: sidebar-cv
classes: wide
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script> 

# Introduction (en)

<p style="text-align:justify;" >
The <b>facial recognition</b> aims to enable the <b>automatic identification of persons</b> from characteristic information extracted from pictures of their face. These techniques have evolved considerably over the last three decades (<a href="https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf">Bromley et al.</a> were already working on a similar subject in 1994), in particular thanks to the contributions of <b>artificial intelligence</b> and in particular <b>deep learning</b>.
<br><br>
<b>Neural networks</b> are now at the heart of many devices and equipment used for the identification of individuals. Their design and integration naturally depend on the intended application and the <b>available hardware resources</b>, as well as other important parameters such as the <b>availability of datasets for their training</b>.
<br><br>
Facial recognition is often approached as a <b>classification problem</b> where it involves determining, using a neural network, the most likely <b>class of belonging</b> of a photo of an individual's face. This approach can, in some cases, be problematic because :<br>
- it requires a fairly substantial <b>set of labeled data</b> potentially tedious to build and update<br>
- the corresponding network must be <b>retrained</b> whenever new classes (new individuals to be identified) need to be added
<br><br>
In cases where, for example, it is desired to recognize new individuals on the fly in a video stream, <b>the classification approach is inappropriate</b> and it is therefore necessary to turn to solutions that require less material resources and computational time.
<br><br>
In these cases, preference will be given to the implementation <b>architectures based on similarity calculation functions</b> which will be used to determine whether or not the photographs of persons to be identified correspond to the representations of known individuals, recorded in a database (and which may itself, if necessary,  be enriched in real time, as new faces are detected). 
<br><br>
We offer you here the description of a solution of this type based on a <b>siamese architecture</b> that we have tested and implemented as part of the <b><a href="https://www.robocup.org/domains/3">RoboCup@Home</a></b>, an international competition in the field of service robotics in which robots must interact with human operators.
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/screen.jpg">
  <figcaption>
  Algorithm outputs
  </figcaption>
</figure>
</center>

<br><br>


# General architecture

<p style="text-align:justify;" >
The facial recognition solution we have developed is based on the integration of tools and neural networks respectively intended to :<br>
- detect the faces of individuals in a photo<br>
- produce, for each isolated face, a <i>identity vector</i> with 64 dimensions representing it<br>
- calculate the distance between the vectors associated with two distinct images<br>
- and determine, by browsing a database, whether the vector associated with one face is close, or not, to that of another already identified
<br><br>
<b>Face detection</b> in an image or video stream, and then their <b>extraction</b>, are performed using tools we'll discuss later.
<br><br>
The heart of the device consists of a model whose objective function calculates a similarity to determine whether or not two face photographs refer to the same individual.
<br><br>
The architecture implemented here is <b>siamese</b> and involves two instances of the same <b>convolutional neural network</b> each taking as input a photograph of the face and providing output a <b>vector representation</b> of it in 64 dimensions.
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
The convolutional network was trained to provide <b>close representations</b>, in Euclidean distance, <b>for two face images of the same person</b> and, conversely, <b>distant or very distant</b> for <b>images of two different persons</b>.
<br><br>
The outputs of the two instances of the network (identical in all points and therefore sharing the same configuration and the same weights) then join and are then used for the calculation of a <b>similarity score directly deduced from the distance separating the vector representations from the images provided as input</b>.
<br><br>
Each face detected in an image or taken from a video stream is then encoded by the network, the resulting vector being <b>compared to a series of known fingerprints</b> stored in a database. The result of this comparison, returned in the form of a scalar value (the similarity score mentioned above), is then evaluated with regard to a threshold beyond which the fingerprints <b>as identical</b> and, consequently, the individual concerned as being <b>identified</b>.
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
The network training was carried out based on the dataset <a href="http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/">VGGFace2</a> by Cao et al. (2018), a publicly accessible dataset containing about 3.3 million images and referring to more than 9000 people.
<br><br>
The images taken from this dataset with great variability in poses, age of subjects, exposures, etc., have been <b>normalized</b> so as to identify the faces and position the characteristic points of these (eyes, nose, mouth) in identical coordinates whatever the cliché considered.
<br><br>
This step of image normalization is critical to network performance. Face detection was performed using a neural network <a href="https://arxiv.org/abs/1905.00641v2">RetinaFace</a> by Deng et al. (2019) to identify an <i>bounding box</i> of the face as well as the characteristic points, the image obtained being <b>cut and transformed</b> so as to position the characteristic points at the predefined positions.
<br><br>
The convolutional network was then trained from these frames.
</p>
<br>

## Architecture

<p style="text-align:justify;" >
The network is built on the basis of an architecture <a href="https://arxiv.org/abs/1905.11946">EfficientNet-B0</a> by Tan and Le (2019), this choice is a compromise between the various constraints of the problem that concerns us since the algorithm will be embedded on the robot, in a graphics card whose capacities are limited.
The number of parameters in memory is constrained and the speed of execution must be sufficient (the decision must be fast because the people to be identified can move, for example).
<br><br>
Relatively short inference times characterize this network (compared to deeper networks, certainly more efficient but inducing significantly longer processing times).
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
We first trained the network on the classification problem of recognizing the photograph of a person among the 9000 identities available. The cost function is then a <b>entropy </b> function (<b><i>crossentropy</i></b>) classical for such a problem.
<br><br>
Once the convergence of the classification problem was achieved, we replaced the last classification layer with a new layer representing the embedding of the image as output.
<br><br>
The previous layers retain the weights of the previous layers from the training in the previous step. This idea is similar to that of <b>transfer learning</b>: intuitively, we try to preserve the characteristics learned during the classification problem and reuse them to build the metric that interests us.
<br><br>
The network was then retrained with an objective function of type <b><i>contrastive</i></b> or <b><i>triplet</i></b> as seen above.
<br><br>
This method makes it possible to quickly train a siamese network. 
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
The facial recognition device was produced by integrating tools and scripts essentially coded in Python.
<br><br>
The neural network itself is implemented using <a href="https://github.com/ pytorch/pytorch">PyTorch</a> by Paszke, Gross, Chintala, Chanan et al. (2016), more precisely in <a href="https://github.com/Lightning-AI/ lightning">Pytorch  Lightning</a> by Falcon et al. (2019), and trained with computational resources from CATIE's <a href="https://www.vaniila.ai/">VANIILA</a> platform.
<br><br>
This made it possible to carry out the successive training sessions in a reasonable time (less than two hours) and the performances obtained appeared quite interesting with an F1 score of 0.92, which is better than the commercial solutions tested.
</p>
<br><br>

# Conclusion

<p style="text-align:justify;" >
We have seen how a first step of extraction and alignment of faces followed, a second of training of a siamese network using an adapted cost function, makes it possible to understand a facial recognition problem.
<br><br>
One of the limitations of this kind of technique, found in other fields, is the need for a very large number of labeled images to train the model. This labelling can be very expensive or impossible. To remedy this, new methods based on self-supervised learning have recently emerged, consisting of training models with a lot of data that does not have a label. We will develop the details of these self-supervised techniques in a future article.
<br><br>

Stay tuned !
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
