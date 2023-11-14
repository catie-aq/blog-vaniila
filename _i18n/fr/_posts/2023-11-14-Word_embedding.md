---
title: "WORD EMBEDDING : QUAND LES MACHINES ARRETENT DE PRENDRE DES RECITS POUR DES TAS DE LETTRES"
tags:
  - NLP
  - word embedding
  - word2vec
  - démonstrateur
excerpt : "Introduction au concept de word embedding <br>- Difficulté : débutant"
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
Pour les machines, comprendre le sens des mots et des phrases est une tâche complexe car elle implique de prendre en compte non seulement la définition des mots, mais également leur connotation, leur contextualisation et leurs relations avec d'autres mots. L'étude de ce problème appartient au domaine du <i>Natural Language Processing</i> (NLP) ou traitement du langage naturel. Un exemple d'application est l'extraction d'informations dans un texte donné, que vous pouvez <a href="https://huggingface.co/spaces/CATIE-AQ/Qamembert">tester librement</a> grâce au modèle entraîné par les experts NLP du CATIE.
<br><br>
Le traitement du langage naturel remonte au début de l'informatique, dans les années 1950. À l'époque, les experts cherchent comment représenter numériquement des mots. Dans les années 2010, la puissance des ordinateurs est telle qu'elle permet la démocratisation des <b>réseaux de neurones</b> ce qui va pousser la représentation vectorielle à s'imposer (à un mot, on associe une séquence de plusieurs centaines de nombres). En effet, la plupart des modèles de <b>machine learning</b> utilisent des <b>vecteurs</b> comme données d'entraînement.
<br><br>
Les modèles de <b>word embedding</b> ont précisément pour fonction de <b>capturer les relations entre les mots d'un corpus de textes et de les traduire en vecteurs</b>. Dans cet article, nous verrons comment interpréter ces vecteurs et comment ils sont générés, en analysant le modèle Word2Vec.
</p>

<br><br>

# L'arithmétique des mots

<p style="text-align:justify;">
Une manière d'interpréter les vecteurs de mots est de les penser comme des coordonnées. En effet, les modèles de <i>word embedding</i> traduisent les relations entre les mots en angles, distances et directions. Par exemple, pour évaluer la <b>proximité sémantique</b> entre 2 mots, il suffit de calculer le cosinus de l'angle entre les 2 vecteurs correspondants : une valeur de 1 (angle de 0°) correspond à des <b>synonymes</b> alors qu'une valeur de -1 indique des <b>antonymes</b> (angle de 180°).
<br><br>
Il est également possible de calculer des relations plus complexes entre les mots. La figure 1 représente la projection de quelques vecteurs de mots dans un espace en 3 dimensions (avant projection, les vecteurs ont des centaines de dimensions). Il y apparaît que le vecteur qui va de <i>reine</i> à <i>roi</i> est à peu près le même que celui qui va de <i>femelle</i> à <i>mâle</i> ou encore <i>jument</i> à <i>étalon</i> <i>i.e.</i> ce vecteur <b>caractérise la relation</b> <i>féminin-masculin</i>. De même, Paris est à la France ce que Berlin est à l'Allemagne, soit :
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
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/vecteurs.svg">
    <figcaption>
    Figure 1 : relations féminin-masculin et pays-capitale
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
Il est possible d'essayer l'arithmétique des mots sur <a href="http://nlp.polytechnique.fr/word2vec">le site de l'école Polytechnique</a>. Il est cependant à noter qu'aucun modèle n'est parfait et que certains résultats d'opérations arithmétiques peuvent être incorrects.
</p>
<!-- TODO: own HF space -->

<br><br>

# Word2Vec

<p style="text-align:justify;">
<b>Word2Vec</b> a été développé par une équipe de chercheurs de Google (Mikolov et al.) en 2013 et est considéré comme étant le modèle qui a permis de <b>démocratiser cette technologie</b>, de par sa simplicité et son efficacité. Même si d'autres modèles de <i>word embedding</i> ont été développés depuis (<b>GloVe</b> et <b>FastText</b> pour ne citer que les plus connus), <b>Word2Vec</b> est encore largement utilisé et cité dans la littérature scientifique.
</p>

<br>

## Quelques définitions

<p style="text-align:justify;">
<b>Contexte</b> : étant donné un texte, le contexte d'un mot est défini comme étant tous les mots dans son <b>voisinage</b>, aux différents endroits du texte où il apparaît. Au voisinage est associée une <b>fenêtre</b> : une fenêtre de taille 3 englobe les 3 mots qui précèdent et les 3 mots qui suivent le mot visé.
<br><br>
<b>Vocabulaire</b> : (sous-)ensemble des mots qui apparaissent dans un texte. Par exemple, dans le texte "La sœur de ma sœur est ma sœur", le vocabulaire associé contiendrait au plus les mots suivants : "la", "sœur", "de", "ma", "est".
<br><br>
<b>Encodage <i>one hot</i></b> : dans un vocabulaire de taille N, l'encodage <i>one hot</i> d'un mot de ce vocabulaire consiste à créer un vecteur de taille N avec N-1 zéros et 1 un correspondant à la position du mot dans le vocabulaire. Par exemple, avec le vocabulaire {"la", "sœur", "de", "ma", "est"}, le vecteur <i>one-hot</i> correspondant à "sœur" est [0, 1, 0, 0, 0].
<br><br>
</p>

<br>

## Fonctionnement

<p style="text-align:justify;">
Le concept de <b>Word2Vec</b> est d'utiliser un <b>réseau de neurones</b> pour résoudre une "<i>fausse tâche</i>", appelée <b>tâche de prétexte</b> : les poids obtenus après entraînement ne servent pas à inférer des résultats mais <b>sont</b> le résultat <i>i.e.</i> les <b>vecteurs de mots</b>. Le modèle se décline en 2 versions (légèrement) différentes : <b>CBOW</b> (pour <i>Continuous Bag Of Words</i>) et <b>Skip Gram</b>. 
<b>CBOW</b> tente de résoudre la tâche qui à un <b>contexte donné associe un mot</b> tandis que <b>Skip Gram</b> fait l'inverse. 
La méthode utilisée étant à peu près la même pour les 2 versions, nous détaillerons par la suite uniquement le modèle <b>Skip Gram</b>.
<br><br>
Pour un texte et une taille de fenêtre donnés, la tâche suivante est définie : soit un mot du texte (l'<i>input</i>), calculer pour chaque autre mot la <b>probabilité qu'il soit dans le contexte de l'<i>input</i></b> (dans la fenêtre). Pour résoudre cette tâche, un réseau de neurones est utilisé; il est composé de :
<ol>
  <li><b>La couche d'<i>input</i></b>; celui-ci est encodé en <b>vecteur <i>one-hot</i></b></li>
  <li><b>Une couche cachée</b>, de taille arbitraire, totalement connectée à l'<i>input</i></li>
  <li><b>La couche d'<i>output</i></b> <i>i.e.</i> un vecteur de probabilité, de la taille du vocabulaire, totalement connectée à la couche cachée</li>
</ol>
Une fonction <b><i>softmax</i></b> est appliquée à l'<i>output</i> afin de n'avoir que des nombres dans l'intervalle [0,1] et dont la somme fait 1.
<br><br>
Par exemple, avec le texte "Les vacances en Nouvelle Aquitaine c'est top, on va au Futuroscope", et une fenêtre de taille 1, la figure 2 illustre comment sont produites les données d'entraînement du modèle :
</p>

<center>
  <figure class="image">
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/fenetre.svg">
    <figcaption>
    Figure 2 : exemple d'<i>inputs</i> et leur contexte
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
Toujours avec le même exemple, la figure 3 représente un réseau de neurones qui est entraîné avec les données précédemment générées.
</p>

<center>
  <figure class="image">
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/reseau.svg">
    <figcaption>
    Figure 3 : exemple de réseau de neurones
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
À la fin, <b>seuls les poids des inputs sont conservés</b> : dans notre cas une matrice de 12 lignes (une ligne par mot) et 3 colonnes (taille de la couche cachée), cf. figure 4. Chaque ligne correspond à un vecteur de mot.
</p>

<center>
  <figure class="image">
    <img src="https://github.com/catie-aq/blog-vaniila/raw/article/word-embedding/assets/images/Word_embedding/matrice.svg">
    <figcaption>
    Figure 4 : Création des vecteurs de mot à partir des poids du modèle
    </figcaption>
  </figure>
</center>

<p style="text-align:justify;">
Il est à noter que dans notre exemple les <i>outputs</i> sont assez prévisibles, car chaque mot n'apparaît qu'une fois dans le texte. Dans la réalité, les corpus de textes utilisés comprennent au moins <b>quelques milliers de mots</b>. Il devrait donc y avoir une probabilité élevée que <i>nouvelle</i> soit dans le voisinage d'<i>aquitaine</i>, ces mots étant souvent associés.
<br><br>
Les vecteurs de mots ainsi produits sont pertinents dans la mesure où <b>2 mots similaires se verront associer 2 vecteurs proches</b>. En toute logique, 2 synonymes devraient effectivement avoir un contexte analogue, ce qui se traduit par 2 <i>outputs</i> quasi égaux pour ces 2 <i>inputs</i>. Le modèle va donc attribuer des poids quasi identiques aux 2 <i>inputs</i>, donnant ainsi 2 vecteurs proches.
</p>

<br><br>

# Applications et limites

<p style="text-align:justify;">
Comme évoqué en introduction, les modèles de <i>word embedding</i> peuvent servir à générer des vecteurs pour <b>entraîner des modèles</b> de NLP plus sophistiqués. Ils peuvent également servir à résoudre des tâches simples, tout en présentant l'avantage d'être <b>peu gourmands en ressources, facilement entraînables et explicables</b>. Il est par exemple possible d'utiliser la similarité entre les mots dans un <b>moteur de recherche</b>, pour remplacer un mot clé par un autre ou étendre la liste des mots clés en piochant dans leur contexte. Grâce aux vecteurs, il est également possible d'étudier la connotation des mots d'un texte pour <b>mettre en évidence des biais</b> liés aux stéréotypes; cf. Garg et al. (2018).
<br><br>
Il existe également des applications de ces modèles en dehors du domaine du traitement du langage. En effet, au lieu de vectoriser des mots avec pour contexte le texte dont ils sont issus, il est par exemple possible de <b>vectoriser les produits d'une <i>marketplace</i></b> avec pour contexte l'historique des achats des utilisateurs, afin de <b>recommander des produits similaires</b>; cf. Grbovic et al. (2015).
<br><br>
La principale limitation de cette technique de vectorisation est qu'elle ne prend pas en compte la <b>polysémie</b> d'un mot : par exemple, dans le texte "L'avocat de la défense mange un avocat", le modèle de <i>word embedding</i> ne créera <b>qu'un seul vecteur</b> pour le mot "avocat". Un autre inconvénient est le <b>travail de prétraitement du corpus</b> à effectuer en amont : il faut définir un vocabulaire <i>i.e.</i> <b>enlever les mots trop répétitifs</b> (ce, de, le...) et potentiellement <b>retirer les formes conjuguées/accordées</b> (est-il souhaitable que "mot" et "mots" aient chacun leur vecteur ?).
<br><br>
Les derniers modèles de langage (GPT, Bloom, Llama...) basés sur des <b><i>transformers</i></b> sont capables de contourner ces limitations. Ils peuvent en effet être <b>directement entraînés sur des textes</b>, sans passer par la définition d'un vocabulaire. Ils utilisent également des vecteurs plus sophistiqués, qui représentent un mot <b>et</b> son contexte, ce qui leur permet de distinguer les différents sens d'un mot.
</p>

<br><br>

# Conclusion
<p style="text-align:justify;">
Pour résumer, les techniques de <i>word embedding</i> ont révolutionné les technologies de NLP, en utilisant des modèles simples, peu coûteux, mais aux résultats impressionnants. Si les <i>transformers</i> remplacent à présent ces modèles dans la plupart des applications, il existe certains cas où ils restent pertinents. Dans un prochain article du blog Vaniila, vous découvrirez une application concrète du <i>word embedding</i>, à travers un projet du CATIE que vous pourrez essayer vous-mêmes !
</p>

<br><br>

# Références
<ul>
  <li><a href="https://arxiv.org/abs/1301.3781">Efficient Estimation of Word Representations in Vector Space</a> de Mikolov et al. (2013),</li>
  <li><a href="http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/">Word2Vec Tutorial - The Skip-Gram Model</a> de McCormick (2016),</li>
  <li><a href="https://doi.org/10.1073/pnas.1720347115">Word embeddings quantify 100 years of gender and ethnic stereotypes</a> de Garg, Schiebinger, Jurafsky et Zou (2018),</li>
  <li><a href="https://arxiv.org/abs/1601.01356">E-commerce in your inbox:
  Product recommendations at scale</a> de Grbovic, Radosavljevic, Djuric, Bhamidipati, Savla, Bhagwan et Sharp (2015)</li>
</ul>

<br><br>

# Commentaires
<script src="https://utteranc.es/client.js"
        repo="catie-aq/blog-vaniila"
        issue-term="pathname"
        label="[Comments]"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
