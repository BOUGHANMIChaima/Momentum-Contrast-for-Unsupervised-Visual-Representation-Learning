# MoCo
Momentum-Contrast-for-Unsupervised-Visual-Representation-Learning: Momentum Contrast (MoCo) comme un moyen de construire de larges dictionnaires cohérents pour l’apprentissage non supervisé avec une perte contrastive.

## Objectif et méthodologie du travail 
Nous supposons qu’il est souhaitable de construire des dictionnaires
qui sont : (i) larges et (ii) consistents
au fur et à mesure qu’ils évoluent pendant l’entrainement. Intuitivement, un dictionnaire

plus large peut mieux échantillonner l’espace visuel continu et de haute dimension sous-
jacent, tandis que les clés du dictionnaire devraient être représentées par le même codeur

ou un codeur similaire afin que leurs comparaisons avec la requête soient cohérentes. Ce-
pendant, les méthodes d’existe qui utilisent des pertes contrastives peuvent être limitées

dans l’un de ces deux aspects.
Nous présentons Momentum Contrast (MoCo) comme un moyen de construire de larges
dictionnaires cohérents pour l’apprentissage non supervisé avec une perte contrastive.

## Description du jeu de données

La base de données MNIST (Modified National Institute of Standards and Technology da-
tabase) de chiffres manuscrits se compose d’un ensemble d’apprentissage de 60 000 exemples

et d’un ensemble de test de 10 000 exemples. Il s’agit d’un sous-ensemble d’un ensemble
plus important disponible auprès du NIST. En outre, les images en noir et blanc du NIST
ont été normalisées en taille et centrées afin de tenir dans une boîte de délimitation de
28x28 pixels et ont subi un traitement anti-alias, ce qui a permis d’introduire des niveaux
de gris.
Pour ce projet, on va prédire des données MNIST avec uniquement 100 labels.

## Baseline : Un modèle supervisé simple

Nous utilisons keras pour construire un réseau neuronal simple qui sera formé sur 100
exemples. Nous testons le modèle sur 10000 exemples de test. la précision plafonne autour
de 73 %

## Description des méthodes utilisées
Article proposé :
L’article étudié est intitulé "Momentum Contrast for Unsupervised Visual Representation
Learning"  publié en 2019 et republié en 2020 par les auteurs Kaiming He, Haoqi Fan,
Yuxin Wu, Saining Xie et Ross présententant une nouvelle méthode appelée Momentum
Contrast ou MoCo, dans laquelle un dictionnaire dynamique est mis en œuvre avec une
file d’attente et un codeur à moyenne mobile. Cela permet de construire un dictionnaire
large et consistant qui facilite l’apprentissage contrastif non supervisé.

## Les principales étapes de l’implémentation de la méthode sont :
### a) Pré-requis : Préformation auto-supervisée et non supervisée pour la vision
par ordinateur :
Traditionnellement, les modèles de vision par ordinateur ont toujours été formés à l’aide
de l’apprentissage supervisé. Cela signifie que les humains ont regardé les images et créé
toutes sortes d’étiquettes pour elles, afin que le modèle puisse apprendre les modèles de ces
étiquettes. Par exemple, un annotateur humain attribuerait une étiquette de classe à une
image ou dessinerait des boîtes de délimitation autour des objets de l’image. Mais comme
le sait toute personne qui a déjà été en contact avec des tâches d’étiquetage, l’effort pour
créer un ensemble de données de formation suffisant est élevé. En revanche, l’apprentissage
auto-supervisé ne nécessite aucune étiquette créée par l’homme. Comme son nom l’indique,

le modèle apprend à se superviser lui-même. En vision par ordinateur, la façon la plus cou-
rante de modéliser cette autosurveillance est de prendre différentes récoltes (crops) d’une

image ou d’y appliquer différentes augmentations et de transmettre les entrées modifiées
dans le modèle. Même si les images contiennent les mêmes informations visuelles mais ne

se ressemblent pas, nous laissons le modèle apprendre que ces images contiennent tou-
jours les mêmes informations visuelles, c’est-à-dire le même objet. Cela conduit le modèle à

apprendre une représentation latente similaire (un vecteur de sortie) pour les mêmes objets.

Nous pourrons ensuite appliquer le transfert-learning sur ce modèle préformé. Habituel-
lement, ces modèles sont ensuite formés sur 10% des données avec des étiquettes pour

effectuer des tâches en aval telles que la détection d’objets et la segmentation sémantique.
### b) Utiliser le contraste Momentum pour l’apprentissage non supervisé :
Dans l’article MoCo, le processus d’apprentissage non supervisé est encadré comme
une recherche de dictionnaire : une clé à chaque vue ou image se voit attribuer, tout comme
dans un dictionnaire. Cette clé est générée en encodant chaque image à l’aide d’un réseau

neuronal convolutionnel, la sortie étant une représentation vectorielle de l’image. Maintenant, si une requête est présentée à ce dictionnaire sous la forme d’une autre image, cette
image de requête est également codée dans une représentation vectorielle et appartiendra à
l’une des clés du dictionnaire, celle ayant la distance la plus basse.

### Data Augmentation
L’augmentation des données dans l’apprentissage contrastif fait le plus souvent appel au
recadrage aléatoire, au sautillement des couleurs et aux flous gaussiens.
Comme les images de l’ensemble de données MNIST sont en niveaux de gris, nous allons
procéder à l’augmentation des données en utilisant un recadrage aléatoire et un flou gaus-
sien.
Nous faisons référence au flou gaussien de l’article SimCLR.
Afin de construire les ensembles de données et les loader de données pour qu’ils incluent les
images augmentées, nous allons utiliser le module torchvision.transforms.
### Définition du modèle
Nous créons les encodeurs (pour la requête et la clé) en utilisant une méthode commune
MNIST ResNet-18 pour le modèle de base. L’étape de propagation vers l’avant fait appel
à la méthode Split Shuffling BatchNorm que nous avons mentionnée précédemment.
Les expériences suggèrent que sans mélanger les BN, les statistiques des sub-batchs peuvent
servir de "signature" pour dire dans quel sub-batch se trouve la clé positive, ce qui entraîne
un overfitting. Le shuffling BN peut supprimer cette signature et éviter une telle manipu-
lation.
### Training
La fonction create_labels crée les labels indiquant les paires positives. Nous gardons éga-
lement la trace des index des paires négatives car la sortie de l’encodeur de clés va être
stockée dans un dictionnaire de file d’attente pour calculer la perte.
L’article original de MoCo v1 utilise une perte asymétrique - une crop est la requête et l’autre
crop est la clé, et elle se propage en retour à une crop (requête). Suite à SimCLR/BYOL, des
options pour une perte symétrique ont été données. Ici, nous utilisons une perte symétrique :
échange des deux cultures et calcul d’une perte supplémentaire.
La fonction de perte est optimisée en utilisant la descente de gradient stochastique momen-
tum. La mise à jour des paramètres est effectuée à l’aide de la fonction copy_param.
Cette démonstration fournit un classificateur kNN sur l’ensemble de test. L’entraînement
d’un classificateur linéaire sur des caractéristiques figées permet souvent d’obtenir une
meilleure précision. L’exécution d’un kNN dans l’espace des caractéristiques peut être consi-
dérée comme une évaluation moins chère mais plus rugueuse que celle d’un classificateur linéaire.

## Résultats 
Le notebook utilise le module pytorch tqdm pour suivre la progression pendant la boucle
d’entrainment.
Nous remarquons que la perte et la précision commencent à plafonner autour de 20 epochs.
Regardons maintenant les graphiques de perte et de précision.
Nous observons rapidement que la précision pour la base de données MNIST est considé-
rablement plus grande que celle obtenue pour d’autres bases de données comme CIFAR10

ou ImageNet
Les résultats sont comparables à ceux que nous avons obtenus à l’aide d’un modèle supervisé.
Nous pouvons ajouter que d’autres améliorations incluent l’adaptation du modèle pour
fonctionner de manière semi-supervisée.


## Références
[1] Philip Bachman, R Devon Hjelm et William Buchwalter. “Learning representa-
tions by maximizing mutual information across views”. In : Advances in neural infor-
mation processing systems 32 (2019).

[2] Ken Chatfield et al. “The devil is in the details : an evaluation of recent feature
encoding methods.” In : BMVC. T. 2. 4. 2011, p. 8.

[3] Ting Chen et al. “A simple framework for contrastive learning of visual representa-
tions”. In : International conference on machine learning. PMLR. 2020, p. 1597-1607.

[4] Adam Coates et Andrew Y Ng. “The importance of encoding versus training with
sparse coding and vector quantization”. In : ICML. 2011.
[5] Jacob Devlin et al. “Bert : Pre-training of deep bidirectional transformers for language
understanding”. In : arXiv preprint arXiv :1810.04805 (2018).

[6] Raia Hadsell, Sumit Chopra et Yann LeCun. “Dimensionality reduction by lear-
ning an invariant mapping”. In : 2006 IEEE Computer Society Conference on Computer

Vision and Pattern Recognition (CVPR’06). T. 2. IEEE. 2006, p. 1735-1742.

[7] Kaiming He et al. “Momentum contrast for unsupervised visual representation lear-
ning”. In : Proceedings of the IEEE/CVF conference on computer vision and pattern

recognition. 2020, p. 9729-9738.
[8] Olivier Henaff. “Data-efficient image recognition with contrastive predictive coding”.
In : International Conference on Machine Learning. PMLR. 2020, p. 4182-4192.

[9] R Devon Hjelm et al. “Learning deep representations by mutual information estima-
tion and maximization”. In : arXiv preprint arXiv :1808.06670 (2018).

[10] Aaron van den Oord, Yazhe Li et Oriol Vinyals. “Representation learning with
contrastive predictive coding”. In : arXiv preprint arXiv :1807.03748 (2018).
[11] Alec Radford et Jeffrey Wu. “Rewon child, david luan, dario amodei, and ilya
sutskever. 2019”. In : Language models are unsupervised multitask learners. OpenAI
Blog 1.8 (2019), p. 9.
[12] Alec Radford et al. “Improving language understanding by generative pre-training”.
In : (2018).
[13] Josef Sivic et Andrew Zisserman. “Video Google : A text retrieval approach to
object matching in videos”. In : Computer Vision, IEEE International Conference on.
T. 3. IEEE Computer Society. 2003, p. 1470-1470.
[14] Yonglong Tian, Dilip Krishnan et Phillip Isola. “Contrastive multiview coding”.
In : European conference on computer vision. Springer. 2020, p. 776-794.

[15] Zhirong Wu et al. “Unsupervised feature learning via non-parametric instance dis-
crimination”. In : Proceedings of the IEEE conference on computer vision and pattern

recognition. 2018, p. 3733-3742.

[16] Chengxu Zhuang, Alex Lin Zhai et Daniel Yamins. “Local aggregation for unsuper-
vised learning of visual embeddings”. In : Proceedings of the IEEE/CVF International

Conference on Computer Vision. 2019, p. 6002-6012.



