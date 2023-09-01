---
title: "Word Embedding: a basic introduction to the world of word vectors"
tags:
  - NLP
  - word embedding
  - word2vec
excerpt : "Introduction to the word embedding concept <br>- Difficuly: beginner"
header:
   overlay_color: "#1C2A4D"
author_profile: false
sidebar:
    nav: sidebar-nlp
classes: wide
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script> 

<!--            Cheat Sheet
paragraphe    <p style="text-align:justify;"></p>
gras          <b>reconnaissance faciale</b>
italique      <i>deep learning</i>
saut de ligne <br><br>
lien externe  <a href="https://example.com">example</a>
-->

# Introduction

<p style="text-align:justify;">
For machines, understanding the meaning of words and sentences is a complex task because it involves taking into account not only the definition of words, but also their connotation, their relationships with other words and the way they interact with the context. The study of this problem belongs to the field of <i>Natural Language Processing</i> (NLP). It serves multiple purposes as, for instance, the extraction of information from a given text, which you can <a href="https://huggingface.co/spaces/CATIE-AQ/Qamembert">test freely</a> using the model trained by the <b>CATIE</b>'s NLP experts.
<br><br>
Natural language processing dates back to the early days of computing, in the 1950s. At the time, experts were looking for ways to represent words digitally. In the 2010s, the power of computers was such that <b>neural networks</b> became popularized, leading to the emergence of <b>vector representation</b> (a word is associated with a sequence of several hundred numbers). Indeed, most <b>machine learning</b> models use vectors as training data.
<br><br>
The aim of <b>word embedding</b> models is precisely to <b>capture the relationships between words</b> in a corpus of texts and translate them into vectors. In this article, we will look at how to interpret these vectors and how they are generated, by analyzing the Word2Vec model.
</p>

# Words Arithmetic

<p style="text-align:justify;">
One way of interpreting word vectors is to think of them as <b>coordinates</b>. Indeed, word embedding models translate the relationships between words into angles, distances and directions. For example, to evaluate the <b>semantic proximity</b> between 2 words, one can simply calculate the cosine of the angle between the 2 corresponding vectors: a value of 1 (angle of 0°) corresponds to <b>synonyms</b>, while a value of -1 indicates <b>antonyms</b> (angle of 180°).
<br><br>
It is also possible to compute more complex relationships between words. Figure 1 shows the projection of some word vectors into a 3-dimensional space (before projection, vectors have hundreds of dimensions). It can be seen that the vector from <i>queen</i> to <i>king</i> is more or less the same as that from <i>female</i> to <i>male</i>, or <i>mare</i> to <i>stallion</i>, that is to say it <b>characterizes the <i>female-male</i> relationship</b>. Similarly, Paris is to France as Berlin is to Germany:
</p>

$$
Paris - France = Berlin - Germany
$$

which is equivalent to:

$$
Paris = Berlin - Germany + France
$$

so one may find Canada's capital city by computing:

$$
Berlin - Germany + Canada
$$

<center>
  <figure class="image">
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/vectors.png">
    <figcaption>
    Figure 1: female-male and country-capital relationships
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
It is possible to try the words arithmetic on <a href="http://nlp.polytechnique.fr/word2vec">the Polytechnique's school website</a>.
</p>
<!-- TODO: own HF space -->

# Word2Vec

<p style="text-align:justify;">
<b>Word2Vec</b> was developed by a team of Google researchers (Mikolov et al.) in 2013 and is considered the model that <b>popularized this technology</b> due to its simplicity and effectiveness. Although other word embedding models have since been developed (GloVe and FastText to name but the best-known), it is still widely used and cited in the scientific literature.
</p>

## Some definitions

<p style="text-align:justify;">
<b>Context</b>: Given a text, the context of a word is defined as <b>all the words in its vicinity</b>, at the various points in the text where it appears. The vicinity is associated with a <b>window</b>: a window of size 3 encompasses the 3 words preceding and the 3 words following the target word.
<br><br>
<b>Vocabulary</b>: (Sub)Set of words that appear in a text. For example, given the text "The sister of my sister is my sister", the associated vocabulary would contain at most the following words: "the", "sister", "of", "my", "is".
<br><br>
<b>One-hot encoding</b>: Given a vocabulary of size N, the one-hot encoding of a word in this vocabulary consists in creating a vector of size N with N-1 zeros and 1 one corresponding to the position of the word in the vocabulary. For example, with the vocabulary {"the", "sister", "of", "my", "is"}, the one-hot vector corresponding to "sister" would be [0, 1, 0, 0, 0].
<br><br>

## The way it works

<p style="text-align:justify;">
The concept behind Word2Vec is to use a <b>neural network</b> to solve a "<i>fake task</i>", known as a <b>pretext task</b>: the weights obtained after training are not used to infer results, but <b>are</b> the result, that is to say the <b>word vectors</b>. The model comes in 2 (slightly) different versions: <b>CBOW</b> (for <i>Continuous Bag Of Words</i>) and <b>Skip Gram</b>. CBOW attempts to solve the task of <b>associating a word with a given context</b>, while Skip Gram does the opposite. As the method used is more or less the same for both versions, we'll only go into detail on the Skip Gram model.
<br><br>
Given a text and a window size, the following task is defined: given a word in the text (the input), <b>compute for each other word the probability that it is in the input's context</b>. To solve this task, a neural network is used, made up of:
<ol>
    <li><b>The input layer</b>, with the word being encoded as a <b>one-hot vector</b></li>
    <li><b>A hidden layer</b>, of arbitrary size, totally connected to the input</li>
    <li><b>The output layer</b>, that is to say a vocabulary-long probability vector, totally connected to the hidden layer</li>
</ol>
A <b>softmax</b> function is applied to the last layer so that the numbers of the output vector all remain in the interval [0,1] and sum to 1.
<br><br>
For example, with the text "Vacations in Nouvelle Aquitaine are dope, we should go to the Futuroscope", and a window of size 1, figure 2 illustrates how the model's training data is produced:
</p>

<center>
  <figure class="image">
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/window.svg">
    <figcaption>
    Figure 2: Inputs and their contexts
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
With the same example, figure 3 represents a neural network trained with the previously generated training data.
</p>

<center>
  <figure class="image">
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/network.svg">
    <figcaption>
    Figure 3: Neural network example
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
After the model was trained, <b>only the input's weights matter</b>: in our case, a 12 row (one by word) and 3 column (size of the hidden layer) matrix, cf figure 4. Each line corresponds to a word vector.
</p>

<center>
  <figure class="image">
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/matrix.svg">
    <figcaption>
    Figure 4: Word vectors retrieval, based on the model weigths
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
Note that in our example, the outputs are fairly predictable, as each word appears only once in the text. In reality, the text corpora used comprise at least <b>a few thousand words</b>. There should therefore be a high probability that <i>nouvelle</i> is in the vicinity of <i>aquitaine</i>, as these words are often associated.
<br><br>
The word vectors thus produced are relevant as <b>2 similar words will be associated with 2 close vectors</b>. Logically, 2 synonyms should indeed have a similar context, which translates into 2 almost equal outputs for these 2 inputs. The model will therefore assign almost identical weights to the 2 inputs, resulting in 2 close vectors.
</p>

# Applications and limits

<p style="text-align:justify;">
As mentioned in the introduction, word embedding models can be used to <b>generate vectors for training more sophisticated NLP models</b>. They can also be used to solve simple tasks, with the advantage of being <b>resource-efficient, easily trainable and explainable</b>. For example, word similarity can be used in a <b>search engine</b> to replace one keyword with another, or to extend the list of keywords based on their context. Thanks to vectors, it is also possible to study the connotation of words in a text to highlight <b>biases linked to stereotypes</b>; cf Garg et al. (2018).
<br><br>
There are also applications of these models outside the field of language processing. Indeed, instead of vectorizing words using the text from which they are derived as a context, it is possible, for example, to <b>vectorize products from a marketplace</b> using users' purchase history as a context, in order to <b>recommend similar products</b>; cf Grbovic et al. (2015).
<br><br>
The main limitation of this vectorization technique is that it does not take into account the <b>polysemy</b> of a word: for example, given the text "The bank is located on the right bank", the word embedding model will only create a <b>single vector</b> for the word "bank". Another drawback is the corpus <b>pre-processing work</b> to be carried out upstream: we need to define a vocabulary, that is to say remove <b>words that are too repetitive</b> (there, is, a...) and potentially <b>remove agreements</b> (is it desirable for "word" and "words" to each have their own vector?).
<br><br>
The latest language models (GPT, Bloom, Llama...) based on <b>transformers</b> are able to overcome these limitations. They can be trained <b>directly on texts</b>, without having to define a vocabulary. They also use more sophisticated vectors, which represent a word <b>and</b> its context, enabling them to distinguish the different meanings of a word.
</p>

# Conclusion
<p style="text-align:justify;">
To sum up, word embedding techniques have revolutionized NLP technologies, using simple, inexpensive models with impressive results. While transformers are gradually replacing these models in most applications, there are some cases where they remain relevant. In a forthcoming article on the Vaniila blog, you'll discover a concrete application of word embedding, through a CATIE project that you will be able to try out for yourself!
</p>

# Références
<ul>
  <li><a href="https://arxiv.org/abs/1301.3781">Efficient Estimation of Word Representations in Vector Space</a> by Mikolov et al. (2013),</li>
  <li><a href="http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/">Word2Vec Tutorial - The Skip-Gram Model</a> by McCormick (2016),</li>
  <li><a href="https://doi.org/10.1073/pnas.1720347115">Word embeddings quantify 100 years of gender and ethnic stereotypes</a> by Garg, Schiebinger, Jurafsky and Zou (2018),</li>
  <li><a href="https://arxiv.org/abs/1601.01356">E-commerce in your inbox:
  Product recommendations at scale</a> by Grbovic, Radosavljevic, Djuric, Bhamidipati, Savla, Bhagwan and Sharp (2015)</li>
</ul>