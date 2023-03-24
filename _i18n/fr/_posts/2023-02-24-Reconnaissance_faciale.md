---
title: "RECONNAISSANCE FACIALE A L’AIDE DE RESEAUX DE NEURONES SIAMOIS"
categories:
  - CV
tags:
  - CV
  - reconnaissance faciale
  - réseaux siamois
  - 2023
excerpt : "CV - Explication des réseaux de neurones siamois pour la reconnaissance faciale <br>Difficulté : débutant"
header:
   overlay_color: "#1C2A4D"
author_profile: false
sidebar:
    nav: sidebar-cv
classes: wide
---

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script> 

# Introduction

<p style="text-align:justify;">
La <b>reconnaissance faciale</b> vise à permettre l'<b>identification automatique de personnes</b> à partir d’informations caractéristiques extraites de photographies de leur visage. Ces techniques ont considérablement évolué au cours des trois dernières décennies (<a href="https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf">Bromley et al.</a> se penchaient déjà sur un sujet similaire en 1994), en particulier grâce aux apports de l’<b>intelligence artificielle</b> et notamment de l’<b>apprentissage profond</b> (<b><i>deep learning</i></b>).
<br><br>
Les <b>réseaux de neurones</b> sont aujourd’hui au cœur de nombreux dispositifs et équipements utilisés pour l’identification d’individus. Leur conception et leur intégration dépendent naturellement de l’application envisagée et des <b>ressources matérielles disponibles</b>, ainsi que d’autres paramètres importants tels que la <b>disponibilité de jeux de données pour leur entraînement</b>.
<br><br>
La reconnaissance faciale est souvent abordée comme un <b>problème de classification</b> où il s’agit de déterminer, à l’aide d’un réseau de neurones, la <b>classe d’appartenance la plus probable</b> de la photographie du visage d’un individu. Cette approche peut, dans certains cas, poser problème car :<br>
- elle nécessite de devoir disposer d’un <b>jeu de données labellisées</b> assez conséquent, potentiellement fastidieux à constituer et à mettre à jour<br>
- le réseau correspondant doit être <b>réentraîné</b> chaque fois que de nouvelles classes (nouveaux individus à identifier) doivent être ajoutées
<br><br>
Dans les cas où l’on souhaite, par exemple, reconnaître à la volée de nouveaux individus dans un flux vidéo, <b>l’approche par classification se révèle inadaptée</b> et il est donc nécessaire de se tourner vers des solutions moins gourmandes en ressources matérielles et en temps de calcul.
<br><br>
Dans ces cas, on privilégiera la mise en œuvre <b>d’architectures prenant appui sur des fonctions de calcul de similarité</b> que l’on utilisera pour déterminer si les photographies de personnes à identifier correspondent, ou pas, aux représentations d’individus connus, enregistrées dans une base de données (et qui pourra elle-même, le cas échéant, être enrichie en temps réel, au fur et à mesure de la détection de nouveaux visages). 
<br><br>
Nous vous proposons ici la description d’une solution de ce type basée sur une <b>architecture siamoise</b> que nous avons notamment testée et mise en œuvre dans le cadre de la <b><a href="https://www.robocup.org/domains/3">RoboCup@Home</a></b>, compétition internationale dans le domaine de la robotique de service dans laquelle les robots doivent interagir avec des opérateurs humains.
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/screen.jpg">
  <figcaption>
  Rendu des sorties de l’algorithme
  </figcaption>
</figure>
</center>

<br><br>


# Architecture générale

<p style="text-align:justify;">
La solution de reconnaissance faciale que nous avons développée repose sur l’intégration d’outils et de réseaux de neurones respectivement destinés à :<br>
- détecter les visages d’individus dans une photographie<br>
  - produire, pour chaque visage isolé, un <i>vecteur d’identité</i> à 64 dimensions le représentant<br>
- calculer la distance entre les vecteurs associés à deux clichés distincts<br>
- et déterminer, en parcourant une base de données, si le vecteur associé à un visage est proche, ou pas, de celui d’un autre déjà identifié
<br><br>
La <b>détection des visages</b> dans une photographie ou un flux vidéo, puis leur <b>extraction</b>, sont effectuées à l’aide d’outils dont nous parlerons plus loin.
<br><br>
Le cœur du dispositif est quant à lui constitué d’un modèle dont la fonction objectif calcule une similarité permettant de déterminer si deux photographies de visage se réfèrent, ou non, à un même individu.
<br><br>
L’architecture mise en œuvre ici est <b>siamoise</b> et fait intervenir deux instances d’un même <b>réseau de neurones convolutif</b> prenant chacun en entrée une photographie de visage et fournissant en sortie une <b>représentation vectorielle</b> de celui-ci en 64 dimensions.
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/overview.png">
  <figcaption>
  Aperçu général de l’architecture du dispositif
  </figcaption>
</figure>
</center>

<p style="text-align:justify;">
Le réseau convolutif a été entraîné de manière à fournir des <b>représentations proches</b>, en distance euclidienne, <b>pour deux clichés de visage de la même personne</b> et, inversement, <b>éloignées ou très éloignées</b> pour les clichés de deux <b>personnes différentes</b>.
<br><br>
Les sorties des deux instances du réseau (identiques en tous points et partageant donc la même configuration et les mêmes poids) se rejoignent ensuite et sont alors utilisées pour le calcul d’un <b>score de similarité directement déduit de la distance séparant les représentations vectorielles des clichés fournis en entrée</b>.
<br><br>
Chaque visage détecté dans une photographie ou tiré d’un flux vidéo est alors encodé par le réseau, le vecteur résultant étant <b>comparé à une série d’empreintes connues</b> stockées dans une base de données. Le résultat de cette comparaison, retourné sous la forme d’une valeur scalaire (le score de similarité évoqué précédemment), est alors évalué au regard d’un seuil au-delà duquel on peut considérer les empreintes <b>comme étant identiques</b> et, par suite, l’individu concerné comme étant <b>identifié</b>.
</p>
<br><br>

  
# Caractéristiques et entraînement du réseau
  
<p style="text-align:justify;">
Le défi consiste ici à concevoir et à entraîner le réseau convolutif de sorte que <b>des entrées similaires soient projetées en des endroits relativement proches dans l’espace des représentations</b> et, inversement, que des <b>entrées différentes soient projetées en des points éloignés</b>.
</p>
<br>

## Jeu de données utilisé et pré-traitements

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/base_de_donnees.png">
  <figcaption>
  Source : 
  <a href="https://paperswithcode.com/dataset/vggface2-1">https://paperswithcode.com/dataset/vggface2-1</a>
  </figcaption>
</figure>
</center>

<p style="text-align:justify;">
L’entraînement du réseau a été réalisé sur la base du jeu de données <a href="http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/">VGGFace2</a> de Cao et al. (2018), un jeu de données accessible publiquement, comportant environ 3,3 millions d’images et se référant à plus de 9000 personnes.
<br><br>
Les images tirées de ce jeu présentant une grande variabilité dans les poses, âge des sujets, expositions, etc., ont été <b>normalisées</b> de manière à identifier les visages et à positionner les points caractéristiques de ceux-ci (yeux, nez, bouche) en des coordonnées identiques quel que soit le cliché considéré.
<br><br>
Cette étape de normalisation des images est critique pour les performances du réseau. La détection des visages a été effectuée à l’aide d’un réseau neuronal <a href="https://arxiv.org/abs/1905.00641v2">RetinaFace</a> de Deng et al. (2019) permettant d’identifier une <i>bounding box</i> du visage ainsi que les points caractéristiques, l’image obtenue étant <b>découpée et transformée</b> de manière à positionner les points caractéristiques aux positions prédéfinies.
<br><br>
Le réseau convolutif positionné au cœur de notre dispositif de reconnaissance faciale a alors été entraîné à partir de ces clichés.
</p>
<br>

## Architecture

<p style="text-align:justify;">
Le réseau est construit sur la base d’une architecture <a href="https://arxiv.org/abs/1905.11946">EfficientNet-B0</a> de Tan et Le (2019), ce choix est un compromis entre les diverses contraintes du problème qui nous occupe puisque l’algorithme sera embarqué sur le robot, dans une carte graphique dont les capacités sont limitées.
Le nombre de paramètres en mémoire est contraint et la vitesse d’exécution doit être suffisante (la décision doit être rapide car les personnes à identifier peuvent se déplacer, par exemple).
<br><br>
Des temps d’inférence relativement courts caractérisent ce réseau (comparativement à des réseaux plus profonds, certes plus performants mais induisant des temps de traitement significativement plus longs).
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/reseau.png">
  <figcaption>
  Architecture du réseau EfficientNet-B0 de Tan et Le (2019)
  </figcaption>
</figure>
</center>

<p style="text-align:justify;">
Remarques :<br>
- le EfficientNet-B0 est le fruit d’un domaine de recherche qui tient une place importante en apprentissage profond : le NAS (<i>Neural Architecture Search</i>), et qui a pour objet d'automatiser et d'optimiser les architectures des réseaux utilisés. Il a donné lieu à de nombreux réseaux, dont les plus populaires sont les <a href="https://arxiv.org/abs/1704.04861">MobileNets</a> de Howard et al. (2017), <a href="https://arxiv.org/abs/1905.11946">EfficientNet</a> (Tan et Le (2019)) ou <a href="https://arxiv.org/abs/2201.03545">ConvNext</a> de Liu et al. (2022).<br>
- de nos jours les <i>transformers</i> pour la vision (<a href="https://arxiv.org/abs/2010.11929">ViT</a> de Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zha et al. (2020)) sont une alternative aux réseaux de neurones convolutifs. On peut citer par exemple le <a href="https://arxiv.org/abs/2103.14030">Swin Transformer</a> de Liu, Lin, Cao, Hu et al. (2021) 
</p>
<br>


## Choix de la fonction objectif

<p style="text-align:justify;">
L’apprentissage de similarités requiert l’utilisation de fonctions objectif appropriées, parmi lesquelles la <i><a href="https://ieeexplore.ieee.org/document/1640964">contrastive loss</a></i> de Hadsell et al. (2005) et la <i><a href="https://arxiv.org/abs/1503.03832">triplet loss</a></i> de Schroff et al. (2015).
<br><br>
La <b><i>contrastive loss</i></b> est définie par :
</p>
$$
L(v_1, v_2)=\frac{1}{2} (1-\alpha)d(v_1, v_2)² + \frac{1}{2} \alpha(max(0,m-d(v_1, v_2)))²
$$

où $$v_1$$ et $$v_2$$ étant deux vecteurs, α est un coefficient qui vaut 1 si les deux vecteurs sont de la même classe, 0 sinon, $$d$$ est une fonction de distance quelconque, et $$m$$ est un réel appelé la marge.
<br><br>
Intuitivement, cette fonction objectif pénalise deux vecteurs de la même classe par leur distance, tandis que deux vecteurs de classes différentes ne sont pénalisés que si leur distance est inférieure à $$m$$.
<br><br>
<p style="text-align:justify;">
La fonction <b>triplet loss</b> fait quant à elle intervenir un troisième vecteur, l’ancre, dans son équation: 
</p>
$$
L(a, v_1, v_2)=max(d(a,v_1)²-d(a,v_2)²+m, 0) 
$$

ici, $$a$$ désigne l’ancre, $$v_1$$ est un vecteur de la même classe que $$a$$ et $$v_2$$ est un vecteur d’une classe différente de $$a$$.
<br><br>
Cette fonction tend simultanément à rapprocher la paire $$(a, v_1)$$  et à éloigner la paire $$(a, v_2)$$ comme présenté sur la figure suivante : 

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/triplet.png">
  <figcaption>
  Triplet loss de Schroff et al. (2015)
  </figcaption>
</figure>
</center>

<p style="text-align:justify;">
De manière générale, l’entraînement des réseaux utilisant directement ces fonctions objectif est assez coûteux, la convergence de ce type de systèmes étant plus longue à obtenir que, par exemple, sur de classiques problèmes de classification.
<br><br>
Afin de contourner cette difficulté, nous avons adopté une approche alternative consistant en un entraînement du réseau en deux étapes.
</p>
<br>

## Entraînement

<p style="text-align:justify;">
Nous avons dans un premier temps entraîné le réseau sur le problème de classification consistant à reconnaître la photographie d’une personne parmi les 9000 identités disponibles. La fonction de coût étant alors une fonction d’<b>entropie croisée</b> (<b><i>crossentropy</i></b>) classique pour un tel problème.
<br><br>
Une fois la convergence du problème de classification obtenue, nous avons remplacé la dernière couche de classification par une nouvelle couche représentant en sortie le plongement de l’image.
<br><br>
Les couches précédentes conservent les poids des couches précédentes issus de l’entraînement à l’étape précédente. Cette idée est similaire à celle de l'<b>apprentissage par transfert</b> (<b><i>transfert learning</i></b>) : intuitivement, on cherche à conserver les caractéristiques apprises lors du problème de classification et à les réutiliser pour construire la métrique qui nous intéresse.
<br><br>
Le réseau a alors été réentraîné avec une fonction objectif de type <b><i>contrastive</i></b> ou <b><i>triplet</i></b> comme vu précédemment.
<br><br>
Cette méthode permet d’entraîner rapidement un réseau siamois. 
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

# Implémentation et intégration

<p style="text-align:justify;">
Le dispositif de reconnaissance faciale été produit par intégration d’outils et de scripts essentiellement codés en langage Python.
<br><br>
Le réseau de neurones est lui-même implémenté à l’aide de <a href="https://github.com/pytorch/pytorch">PyTorch</a> de Paszke, Gross, Chintala, Chanan et al. (2016), plus précisément en <a href="https://github.com/Lightning-AI/lightning">Pytorch Lightning</a> de Falcon et al. (2019), et entraîné avec les ressources de calcul de la plateforme <a href="https://www.vaniila.ai/">VANIILA</a> du CATIE.
<br><br>
Cela a permis de réaliser les entraînements successifs en un temps raisonnable (moins de deux heures) et les performance obtenues sont apparues tout à fait intéressantes avec un score F1 de 0,92, ce qui est meilleur que les solutions du commerce testées.
</p>
<br><br>

# Conclusion

<p style="text-align:justify;">
Nous avons vu comment une première étape d’extraction et d’alignement des visages suivie, d’une seconde d’entraînement d’un réseau siamois à l’aide d’une fonction de coût adaptée, permet d’appréhender une problématique de reconnaissance faciale.
<br><br>
Une des limites de ce genre de techniques, trouvables dans d’autres domaines, est la nécessité d’un très grand nombre d’images étiquetées pour entraîner le modèle. Cet étiquetage peut être très coûteux voire impossible. Pour remédier à cela, de nouvelles méthodes basées sur l’apprentissage auto-supervisé sont apparues récemment, consistant à entraîner les modèles avec de nombreuses données qui n’ont pas d’étiquette. Nous développerons les détails de ces techniques auto-supervisées dans un prochain article.
<br><br>
Stay tuned !
</p>

<center>
<figure class="image">
  <img src="https://raw.githubusercontent.com/catie-aq/blog-vaniila/main/assets/images/Reconnaissance_faciale/epock.jpg">
  <figcaption>
  Epock, le robot du CATIE, pendant la RoboCup 2019
  </figcaption>
</figure>
</center>


<br><br>

# Références

<p style="text-align:justify;">
- <a href="https://arxiv.org/abs/2201.03545">A ConvNet for the 2020s</a> de Liu et al. (2022)<br>
- <a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</a> de Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zha et al. (2020)<br>
- <a href="https://ieeexplore.ieee.org/document/1640964">Dimensionality Reduction by Learning an Invariant Mapping</a> de Hadsell et al. (2005)<br>
- <a href="https://arxiv.org/abs/1905.11946">EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</a> de Tan et Le (2019)<br>
- <a href="https://arxiv.org/abs/1503.03832">FaceNet: A Unified Embedding for Face Recognition and Clustering</a> de Schroff et al. (2015)<br>
- <a href="https://arxiv.org/abs/1704.04861">MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications</a> de Howard et al. (2017)<br>
- <a href="https://github.com/pytorch/pytorch">PyTorch</a> de Paszke, Gross, Chintala, Chanan et al. (2016)<br>
- <a href="https://github.com/Lightning-AI/lightning">Pytorch Lightning</a> de Falcon et al. (2019)<br>
- <a href="https://arxiv.org/abs/1905.00641v2">RetinaFace: Single-stage Dense Face Localisation in the Wild</a> de Deng et al. (2019)<br>
- <a href="https://proceedings.neurips.cc/paper/1993/file/288cc0ff022877bd3df94bc9360b9c5d-Paper.pdf">Signature Verification using a "Siamese" Time Delay Neural Network</a> de Bromley et al. (1994)<br>
- <a href="https://arxiv.org/abs/2103.14030">Swin Transformer: Hierarchical Vision Transformer using Shifted Windows</a> de Liu, Lin, Cao, Hu et al. (2021)<br>
- <a href="https://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf">VGGFace2: A dataset for recognising faces across pose and age</a> de Cao et al. (2018)
</p>

<br><br>

# Commentaires
<script src="https://utteranc.es/client.js"
        repo="catie-aq/blog-vaniila"
        issue-term="pathname"
        label="[Commentaires]"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
