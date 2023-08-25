---
title: "Word Embedding : quand les machines arrêtent de prendre des récits pour des tas de lettres"
tags:
  - NLP
  - word embedding
  - word2vec
  - glove
  - démonstrateur
excerpt : "Introduction au concept de word embedding <br>- Difficulté : débutant"
header:
   overlay_color: "#1C2A4D"
author_profile: false
sidebar:
    nav: sidebar-cv
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
Pour les machines, comprendre le sens des mots et des phrases est une tâche complexe car elle implique de prendre en compte non seulement la définition des mots, mais aussi leur connotation, leur contextualisation et leurs relations avec d'autres mots. L'étude de ce problème appartient au domaine du <i>Natural Language Processing</i> (NLP) ou traitement du langage naturel. Un exemple d'application est l'extraction d'informations dans un texte donné, que vous pouvez <a href="https://huggingface.co/spaces/CATIE-AQ/Qamembert">tester librement</a> grâce au modèle entraîné par les experts NLP du CATIE.
<br><br>
Le traitement du langage naturel remonte au début de l'informatique, dans les années 50. À l'époque, les experts cherchent comment représenter numériquement des mots. Dans les années 2010, la puissance des ordinateurs est telle qu'elle permet la démocratisation des <b>réseaux de neurones</b> ce qui va pousser la représentation vectorielle à s'imposer (à un mot on associe une séquence de plusieurs centaines de nombres). En effet la plupart des modèles de <b>machine learning</b> utilisent des <b>vecteurs</b> comme données d'entraînement.
<br><br>
Les modèles de <b>word embedding</b> ont précisément pour fonction de <b>capturer les relations entre les mots d'un corpus de textes et de les traduire en vecteurs</b>. Dans cet article, nous verrons comment interpréter ces vecteurs et comment ils sont générés, en analysant les modèles Word2Vec et GloVe.
</p>

# L'arithmétique des mots

<p style="text-align:justify;">
Une manière d'interpréter les vecteurs de mots est de les penser comme des coordonnées. En effet, les modèles de word embedding traduisent les relations entre les mots en angles, distances et directions. Par exemple, pour évaluer la <b>proximité sémantique</b> entre 2 mots, il suffit de calculer le cosinus de l'angle entre les 2 vecteurs correspondants : une valeur de 1 (angle de 0°) correspond à des <b>synonymes</b> alors qu'une valeur de -1 indique des <b>antonymes</b> (angle de 180°).
<br><br>
Il est également possible de calculer des relations plus complexes entre les mots. La figure 1 représente la projection de quelques vecteurs de mots dans un espace en 3 dimensions (avant projection, les vecteurs ont des centaines de dimensions). Il y apparaît que le vecteur qui va de <i>reine</i> à <i>roi</i> est à peu près le même que celui qui va de <i>femelle</i> à <i>mâle</i> ou encore <i>jument</i> à <i>étalon</i> <i>ie</i> ce vecteur <b>caractérise la relation</b> <i>féminin-masculin</i>. De même, Paris est à la France ce que Berlin est à l'Allemagne, soit :
</p>

$$
Paris - France = Berlin - Allemagne
$$

ce qui est équivalent à

$$
Paris = Berlin - Allemagne + France
$$

et il est donc possible de retrouver la capitale du Canada en calculant

$$
Berlin - Allemagne + Canada
$$

<center>
  <figure class="image">
    <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Word_embedding/vecteurs.png">
    <figcaption>
    Figure 1 : Relation féminin-masculin et pays-capitale
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
Il est possible d'essayer l'arithmétique des mots sur <a href="http://nlp.polytechnique.fr/word2vec">le site de l'école Polytechnique</a>.
</p>

# Les modèles de Word Embedding

<p style="text-align:justify;">
Il existe de nombreux modèles de word embedding mais 2 d'entre eux sont particulièrement cités dans la littérature scientifique : <b>Word2Vec</b> et <b>GloVe</b>. Le premier a été développé par une équipe de chercheurs de Google en 2013<sup><a href="#bib:original_word2vec">[1]</a></sup> et est considéré comme étant le modèle qui a permis de <b>démocratiser cette technologie</b>. Le second a été publié par l'université de Stanford en 2014<sup><a href="#bib:glove">[2]</a></sup> et se positionne comme le successeur de Word2Vec.
</p>

## Quelques définitions

<p style="text-align:justify;">
<b>Contexte</b> : Étant donné un texte, le contexte d'un mot est défini comme étant tous les mots dans son <b>voisinage</b>, aux différents endroits du texte où il apparaît. Au voisinage est associée une <b>fenêtre</b> : une fenêtre de taille 3 englobe les 3 mots qui précèdent et les 3 mots qui suivent le mot visé.
<br><br>
<b>Vocabulaire</b> : (Sous-)Ensemble des mots qui apparaissent dans un texte. Par exemple, étant donné le texte "La soeur de ma soeur est ma soeur", le vocabulaire associé contiendrait au plus les mots suivant : "la", "soeur", "de", "ma", "est".
<br><br>
<b>Encodage <i>one hot</i></b> : Étant donné un vocabulaire de taille N, l'encodage one hot d'un mot de ce vocabulaire consiste à créer un vecteur de taille N avec N-1 zéros et 1 un correspondant à la position du mot dans le vocabulaire. Par exemple, avec le vocabulaire {"la", "soeur", "de", "ma", "est"}, le vecteur one-hot correspondant à "soeur" est [0, 1, 0, 0, 0].
<br><br>
<b>Matrice de co-occurence</b> : Étant donné un texte, un vocabulaire et une taille de fenêtre, une matrice de co-occurence est une matrice dont chaque ligne et chaque colonne correspondent à un mot du vocabulaire, et l'élément à l'intersection des mots <i>mot1</i> et <i>mot2</i> correspond au nombre de fois où <i>mot2</i> est dans le voisinage de <i>mot1</i> (selon la taille de fenêtre). Par exemple, avec la phrase "La soeur de ma soeur est ma soeur" et une fenêtre de taille 1, cela donne :

| |la|soeur|de|ma|est|
|-|-|-|-|-|-|
|la|0|1|0|0|0|
|soeur|1|0|1|2|1|
|de|0|1|0|1|0|
|ma|0|2|1|0|1|
|est|0|1|0|1|0|

</p>

## Word2Vec

<p style="text-align:justify;">
Le concept de Word2Vec est d'utiliser un <b>réseau de neurones</b> pour résoudre une "<i>fausse tâche</i>", appelée <b>tâche de prétexte</b> : les poids obtenus après entraînement ne servent pas à inférer des résultats mais <b>sont</b> le résultat <i>ie</i> les <b>vecteurs de mots</b>. Le modèle se décline en 2 versions (légèrement) différentes : <b>CBOW</b> (pour <i>Continuous Bag Of Words</i>) et <b>Skip Gram</b>. CBOW tente de résoudre la tâche qui à un <b>contexte</b> donné associe un <b>mot</b> tandis que Skip Gram fait l'inverse. La méthode utilisée étant à peu près la même pour les 2 versions, nous détaillerons par la suite uniquement le modèle Skip Gram.
<br><br>
Étant donnés un texte et une taille de fenêtre, la tâche suivante est définie : soit un mot du texte (l'input), calculer pour chaque autre mot la <b>probabilité qu'il soit dans le voisinage de l'input</b> (dans la fenêtre). Pour résoudre cette tâche, un réseau de neurones est utilisé; il est composé de :
<ol>
  <li>La couche d'input; celui-ci est encodé en vecteur one-hot</li>
  <li>Une couche cachée, de taille arbitraire, totalement connectée à l'input</li>
  <li>La couche d'output <i>ie</i> un vecteur de probabilité, de la taille du vocabulaire, totalement connectée à la couche cachée</li>
</ol>
Une fonction <b>softmax</b> est appliquée à l'output afin de n'avoir que des nombres dans l'intervalle [0,1] et dont la somme fait 1.
<br><br>
Par exemple, avec le texte "Les vacances en Nouvelle Aquitaine c'est top, on va au Futuroscope", et une fenêtre de taille 1, la figure 2 illustre comment sont produites les données d'entraînement du modèle :
</p>

<center>
  <figure class="image">
    <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Word_embedding/fenetre.svg">
    <figcaption>
    Figure 2 : Exemple d'inputs et leur contexte
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
Toujours avec le même exemple, la figure 3 représente un réseau de neurones qui est entraîné avec les données précédemment générées.
</p>

<center>
  <figure class="image">
    <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Word_embedding/reseau.svg">
    <figcaption>
    Figure 3 : Exemple de réseau de neurones
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
À la fin, seuls les poids des inputs sont conservés : dans notre cas une matrice de 12 lignes (une ligne par mot) et 3 colonnes (taille de la couche cachée), cf figure 4. Chaque ligne correspond à un vecteur de mot.
</p>

<center>
  <figure class="image">
    <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Word_embedding/matrice.svg">
    <figcaption>
    Figure 4 : Création des vecteurs de mot à partir des poids du modèle
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
Il est à noter que dans notre exemple les outputs sont assez prévisibles, car chaque mot n'apparaît qu'une fois dans le texte. Dans la réalité, les corpus de textes utilisés comprennent au moins <b>quelques milliers de mots</b>. Il devrait donc y avoir une probabilité élevée que <i>nouvelle</i> soit dans le voisinage d'<i>aquitaine</i>, ces mots étant souvent associés.
<br><br>
Les vecteurs de mots ainsi produits sont pertinents dans la mesure où <b>2 mots similaires se verront associer 2 vecteurs proches</b>. En toute logique, 2 synonymes devraient effectivement avoir un contexte analogue, ce qui se traduit par 2 outputs quasi égaux pour ces 2 inputs. Le modèle va donc attribuer des poids quasi identiques aux 2 inputs, donnant ainsi 2 vecteurs proches.
</p>

## Glove
<p style="text-align:justify;">
Word2Vec construit ses vecteurs en se basant sur un contexte local (à l'intérieur de la fenêtre). Avant lui, d'autres algorithmes tiraient partie de la structure globale du corpus, avec des matrices qui comptaient le nombre d'occurence d'un mot dans chaque document du corpus. GloVe propose une méthode à mi-chemin entre les 2, avec une matrices de co-occurence. La matrice obtenue à partir du corpus permet de calculer la probabilité qu'un mot soit dans le contexte d'un autre. 
<br><br>
L'idée derrière GloVe est la suivante : soit Pki la probabilité que le mot k soit dans le contexte du mot i, alors le rapport Pki/Pkj nous permet d'apprécier la relation entre le mot k et les mots i et j. En effet, prenons i = guidon et j = gouvernail; avec k = vélo, le rapport devrait être relativement grand. À l'inverse, k = bateau donnerait une petite valeur. Quant à des mots proches ou éloignés de i et de j, tels k = véhicule ou k = banane, ils donneraient un rapport d'à peu près 1.
<br><br>
Ainsi les vecteurs sont construits de telle sorte qu'en calculant le produit scalaire entre 2 d'entre eux, il est possible de retrouver la probabilité que le mot associé au premier vecteur soit dans le contexte du mot associé au deuxième. Pour ce faire une fonction de coût est définie de la façon suivante :

où F rend compte de la distance entre le produit scalaire et la probabilité de voisinage. Un algorithme d'optimisation (gradient descent) permet enfin d'ajuster les vecteurs pour réduire la valeur de la fonction de coût.
</p>

# Applications et limites

<p style="text-align:justify;">
Comme évoqué en introduction, les modèles de word embedding peuvent servir à générer des vecteurs pour <b>entraîner des modèles</b> de NLP plus sophistiqués. Ils peuvent également servir à résoudre des tâches simples, tout en présentant l'avantage d'être <b>peu gourmands en ressources, facilement entraînables et explicables</b>. Il est par exemple possible d'utiliser la similarité entre les mots dans un <b>moteur de recherche</b>, pour remplacer un mot clé par un autre ou étendre la liste des mots clés en piochant dans leur contexte. Grâce aux vecteurs, il est également possible d'étudier la connotation des mots d'un texte pour <b>mettre en évidence des biais</b> liés aux stéréotypes.
<br><br>
Il existe également des applications de ces modèles en dehors du domaine du traitement du langage. En effet, au lieu de vectoriser des mots avec pour contexte le texte dont ils sont issus, il est par exemple possible de <b>vectoriser les produits d'une <i>marketplace</i></b> avec pour contexte une séquence de clics d'un utilisateur, afin de <b>recommander des produits similaires</b>.
<br><br>

<!--
Limitation 2 : accords de genre / conjugaison = mots différents (vecteurs différents) => prétraitement (lemmatisation)
Fasttext: intermédiaire entre word2vec et transformer (gros modèle de word2vec)
+ Glove en conclusion
-->

La principale limitation de cette technique de vectorisation est qu'elle ne prend pas en compte la <b>polysémie</b> d'un mot. Étant donné le texte "L'avocat de la défense mange un avocat", le modèle de word embedding ne créera <b>qu'un seul vecteur</b> pour le mot "avocat". C'est de l'<b>embedding statique</b>. Les derniers modèles de langage (GPT, Bloom, Llama...) basés sur des <b><i>transformers</i></b> utilisent des vecteurs plus sophistiqués, qui représentent un mot <b>et</b> son contexte.
</p>

# Références
<ol>
  <li id="bib:original_word2vec">Mikolov, Tomas; et al. (2013). "Efficient Estimation of Word Representations in Vector Space". <a href="https://arxiv.org/abs/1301.3781">arXiv:1301.3781</a></li>
  <li id="bib:glove">Jeffrey Pennington, Richard Socher, and Christopher D. Manning. (2014). <a href="https://nlp.stanford.edu/pubs/glove.pdf">GloVe: Global Vectors for Word Representation</a>.</li>
  <li id="bib:word2vec_tutorial">McCormick, C. (2016, April 19). Word2Vec Tutorial - The Skip-Gram Model. Retrieved from <a href="http://www.mccormickml.com">http://www.mccormickml.com</a></li>
</ol>
