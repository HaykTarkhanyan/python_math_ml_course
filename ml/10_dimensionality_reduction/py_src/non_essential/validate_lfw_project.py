"""Answer key for Project 2 ("When eigenfaces meet the real world") in 10_dimensionality_reduction.qmd.

Run before changing the assignment, to confirm the arc it asks students to discover is
still real. Results as of 2026-08-16 (sklearn 1.7.1, seed 509, 25% stratified test split,
LFW min_faces_per_person=70 resize=0.4 -> 1288 images, 7 people, 50x37):

    Olivetti  PCA(150) + 1-NN ......... 0.920     <- the Project 1 number
    LFW       PCA(150) + 1-NN ......... 0.571     <- Part A: the collapse
    majority-class baseline ........... 0.411     (George W Bush, 530/1288 = 41.1%)

    corr(PC1 score, image mean brightness) = +0.998, and PC1 is 20.4% of all variance.
    PC2..PC5 correlate at |r| < 0.05. So the single fattest direction in the data is
    literally "how bright is this photograph" -- this is Part B's payoff and it is about
    as clean a demonstration of "PCA is unsupervised" as exists.

    Part C fixes:
      drop first 1 PC ................. 0.637
      drop first 2 PCs ................ 0.640
      drop first 3 PCs ................ 0.655     <- the classic Fisherfaces-paper trick
      drop first 5 PCs ................ 0.652
      standardize pixels first ........ 0.575     <- barely moves; the intended disappointment
      Fisherfaces (PCA -> LDA) ........ 0.820     <- the real win, +25 points
      drop 3 + LDA .................... 0.820     <- no further gain

Two teaching points fall out of that table and both are deliberate in the assignment:
  * Standardizing pixels does NOT substitute for dropping the leading components. It
    equalizes each *pixel* across the dataset; it does not remove a per-*image* brightness
    offset, which is what PC1 encodes.
  * "drop 3 + LDA" equals plain LDA. Once you use the labels, hand-deleting the brightness
    axis is redundant -- LDA already gives it no weight, because it does not separate
    classes. Task 10 asks students to notice exactly this.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import numpy as np
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 509

lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X, y, names = lfw.data, lfw.target, lfw.target_names
print(f"LFW: {X.shape}, {len(names)} people, image {lfw.images.shape[1:]}")
counts = np.bincount(y)
for n, c in sorted(zip(names, counts), key=lambda t: -t[1]):
    print(f"  {n:24s} {c:4d}  ({c/len(y)*100:.1f}%)")
maj = counts.max() / len(y)
print(f"MAJORITY-CLASS BASELINE: {maj:.3f}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)


def knn_acc(tr, te, k=150, drop=0, scale=False, lda=False):
    steps = []
    if scale:
        steps.append(StandardScaler())
    p = PCA(n_components=k, random_state=SEED)
    A, B = tr, te
    if scale:
        s = StandardScaler().fit(tr)
        A, B = s.transform(tr), s.transform(te)
    p.fit(A)
    Ztr, Zte = p.transform(A)[:, drop:], p.transform(B)[:, drop:]
    if lda:
        L = LinearDiscriminantAnalysis().fit(Ztr, ytr)
        Ztr, Zte = L.transform(Ztr), L.transform(Zte)
    m = KNeighborsClassifier(n_neighbors=1).fit(Ztr, ytr)
    return m.score(Zte, yte)


# Olivetti reference (the Project 1 number)
o = fetch_olivetti_faces(shuffle=True, random_state=SEED)
oXtr, oXte, oytr, oyte = train_test_split(o.data, o.target, test_size=0.25,
                                          stratify=o.target, random_state=SEED)
op = PCA(n_components=150, random_state=SEED).fit(oXtr)
om = KNeighborsClassifier(n_neighbors=1).fit(op.transform(oXtr), oytr)
print(f"\nOlivetti  PCA(150) + 1-NN : {om.score(op.transform(oXte), oyte):.3f}")

base = knn_acc(Xtr, Xte)
print(f"LFW       PCA(150) + 1-NN : {base:.3f}")

# Is PC1 measuring brightness?
p = PCA(n_components=10, random_state=SEED).fit(Xtr)
Z = p.transform(Xtr)
bright = Xtr.mean(axis=1)
print("\ncorr(PC score, image mean brightness):")
for i in range(5):
    print(f"  PC{i+1}: {np.corrcoef(Z[:, i], bright)[0,1]:+.3f}   "
          f"(evr {p.explained_variance_ratio_[i]*100:.1f}%)")

print("\nfixes:")
for d in (0, 1, 2, 3, 5):
    print(f"  drop first {d} PCs        : {knn_acc(Xtr, Xte, drop=d):.3f}")
print(f"  standardize pixels first : {knn_acc(Xtr, Xte, scale=True):.3f}")
print(f"  Fisherfaces (PCA->LDA)   : {knn_acc(Xtr, Xte, lda=True):.3f}")
print(f"  drop 3 + LDA             : {knn_acc(Xtr, Xte, drop=3, lda=True):.3f}")
