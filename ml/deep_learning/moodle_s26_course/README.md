# Deep Learning [s26] — LMU Munich Moodle archive (Rezaei, Rügamer)

Archived from LMU Moodle course **s26_dl** (id=44585), Summer Semester 2026.
Source: <https://moodle.lmu.de/course/view.php?id=44585>
Archived: 2026-07-12

> This is a snapshot of the **live LMU Moodle course** — all slide PDFs, the mock exam,
> and the full lab-materials folder (R + Python, questions + solutions).
> The sibling folder `../` holds separate I2DL reference/planning docs
> (`i2dl_slides_summary.md`, `i2dl_course_outline.md`, `additional_resources.md`).

---

## Course description

In recent years, deep learning has steadily increased in popularity, mainly due to
its state-of-the-art performance in image and speech recognition, text mining, and
related tasks. Deep neural networks attempt to automatically learn multi-level
representations and features of data and are able to uncover complex underlying data
structures.

The lecture aims at providing a basic theoretical and practical understanding of
modern neural network approaches. It starts with the necessary background on
traditional artificial neural networks, backpropagation, online learning, and
regularization, then covers special deep-learning methods such as dropout and
rectified linear units, and finally more advanced topics including convolutional
layers, recurrent neural networks, and autoencoders — plus practical applications
and open-source deep learning libraries.

## People

**Lecture:** David Rügamer (DR), Mina Rezaei (MR)
**Lab session:** Emanuel Sommer, Julius Kobialka, Sarah Deubner (SD)

## Time & location

| | When | Where |
|---|---|---|
| **Lecture** | Every Monday, 16:00–18:00 (c.t.) | Geschw.-Scholl-Pl. 1 (M) — M 010 |
| **Lab** | Every Thursday, 10:00–12:00 (c.t.) | Geschw.-Scholl-Pl. 1 (A) — A125 |

## Exam

- **Duration:** 90 minutes
- **Modality:** Closed book. Only a single (digitally written and then printed)
  handwritten A4 sheet is allowed as assistance; both sides may be used.
- **Main exam:** 22.07.2026, starting 16:00 — B201, main building
- **Retake:** 28.09.2026, starting 12:00 — Kaulbachstr. 37, Room 023
- **Mock exam:** `mock_exam/MockExamDL2026.pdf` (an exam of similar *style*, not
  necessarily content, to understand the type of questions)

## Requirements

- English
- Statistics and Data Science Master
- Background in Machine Learning (preferably *Fortgeschrittene Computerintensive
  Methoden* / Computational Methods II, or Predictive Modeling)
- Background in optimization (e.g. *Computerintensive Methoden* / Computational
  Methods I in the statistics master)
- Solid programming knowledge in R or Python

*Note:* External people are welcome subject to space/resources; they must be aware
of the prerequisites and arrange exam details with their Prüfungsamt themselves.

## Questions & communication

- Lecture/exercise questions: use the corresponding Moodle forum or the Etherpad.
- Slide mistakes/typos: open a well-explained issue or (better) a pull request on GitHub.
- Course-organizational questions: email **all** of:
  david.ruegamer@stat.uni-muenchen.de; mina.rezaei@stat.uni-muenchen.de; emanuel@stat.uni-muenchen.de

Forums:
- Announcements — <https://moodle.lmu.de/mod/forum/view.php?id=2428774>
- Forum for Questions — <https://moodle.lmu.de/mod/forum/view.php?id=2428775>
- Student Discussion Forum (Etherpad Lite) — <https://moodle.lmu.de/mod/etherpadlite/view.php?id=2428776>

## Evaluation

Evaluate lecture and labs at
<https://www.lehrevaluation.uni-muenchen.de/evasys/public/online/index>
(open from 8 June until **19.06.2026**). TANs:

| Instructor | Role | TAN |
|---|---|---|
| David Rügamer | Lecture | RE6YK |
| Mina Rezaei | Lecture | 773XQ |
| Sarah Deubner | Lab | 8HPQ8 |
| Julius Kobialka | Lab | V4PCA |
| Emanuel Sommer | Lab | 2TXWN |

---

## Schedule (as published; dynamic during the semester)

Lab programming language for the week is shown in **bold**.

| Week (starting) | Lecture (Monday) | Lab (Thursday) |
|---|---|---|
| 16 (13.4.2026) | DR: Introduction / History / Single Neuron / XOR / Single Hidden | Lab 1: Introduction, MLPs, XOR, Hyperplanes (**R**) |
| 17 (20.4.2026) | DR: Matrix Notation, Multilayer, Multiclass, Univ. Approx. | Lab 2: GD, Chain Rule, Comp Graph (**R**) |
| 18 (27.4.2026) | DR: Basic Training, Comp. Graph, Backprop 1 & 2, Hard-/Software | Lab 3: Backprop (**R**) |
| 19 (4.5.2026) | DR (online): Basic Regularization, Early Stopping, Dropout & Augmentation | Lab 4: Autodiff, L2 reg. (**Python**) |
| 20 (11.5.2026) | DR: Challenges, Adv. Optim, Initialization, Activations | Public Holiday (No Class) |
| 21 (18.5.2026) | DR: Buffer | Lab 5: Regularization, Convergence (**Python**) |
| 22 (25.5.2026) | Public Holiday (No Class) | Lab 6: CNNs (intro and math) (**Python**) |
| 23 (1.6.2026) | MR: Intro to CNNs, Conv2d, CNN Properties, Components, Conv vs Cross-Corr, Application | Public Holiday (No Class) |
| 24 (8.6.2026) | MR: Conv-1D/2D/3D, Dilated & Transposed, Separable & Flattened, Modern CNN I & II | Lab 7: CNNs continued (**Python**) |
| 25 (15.6.2026) | MR: Intro to RNN & Comp Graph, Backprop Through Time, Modern RNN, Applications | Lab 8: RNNs (**Python**) |
| 26 (22.6.2026) | MR: Attention & Transformers, Unsupervised Learning, Manifold Learning, AutoEncoders, Regularized AE, Specific AE | Lab 9: BOW, LSTM, ATT (**Python**) |
| 27 (29.6.2026) | MR: VAE | Lab 10 (by SD): DAE, KLD, VAE (**Python**) |
| 28 (6.7.2026) | No class | No class |
| 29 (13.7.2026) | Q&A (online) | Q&A (online) |

### Online-lecture videos (Week 19, external Google Drive — not downloaded)

- Basic Regularization — <https://drive.google.com/file/d/1JnkQCEJyhBWBygyDe52_jd9trrSbqn46/view?usp=sharing>
- Early Stopping — <https://drive.google.com/file/d/1CirLGai9zOqmNdNWL54BYgpxlRWO57XO/view?usp=sharing>
- Dropout and Augmentation — <https://drive.google.com/file/d/1D2ht1cl5IdH4DvdvmE8Njnjkq63WHoUy/view?usp=sharing>

Week-19 also references I2ML web chapters (not PDFs):
Ridge Regularization, Regularization in Non-Linear Models, Geometric Interpretation L2
— <https://slds-lmu.github.io/i2ml/chapters/15_regularization/>

---

## Literature & references

Marked (*) = recommended.

- (*) Deep Learning: Foundations and Concepts (Bishop 2025) — <https://www.bishopbook.com/>
- (*) Dive into Deep Learning — <https://d2l.ai/>
- (*) Goodfellow et al. (2016) Deep Learning — <http://www.deeplearningbook.org/>
- (*) Probabilistic ML: An Introduction (Murphy 2022) — <https://probml.github.io/pml-book/book1.html>
- Probabilistic ML: Advanced Topics (Murphy 2023) — <https://probml.github.io/pml-book/book2.html>
- Bishop (1995) Neural Networks for Pattern Recognition — <https://dl.acm.org/citation.cfm?id=525960>
- Chollet (2017) Deep Learning with Python — <https://www.manning.com/books/deep-learning-with-python>
- Chollet (2018) Deep Learning with R — <https://www.manning.com/books/deep-learning-with-r>
- Ghatak (2019) Deep Learning with R — <https://www.springer.com/de/book/9789811358494>
- Regularization for Deep Learning: A Taxonomy — <https://arxiv.org/pdf/1710.10686.pdf>
- Understanding LSTM and its diagrams — <https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714>
- The matrix calculus you need for deep learning — <https://explained.ai/matrix-calculus/>
- Tutorial on Variational Autoencoders — <https://arxiv.org/pdf/1606.05908.pdf>

---

## Local folder structure

```
moodle_s26_course/
├── README.md                      # this file
├── slides/                        # 43 lecture slide decks, by week
│   ├── week16_intro_mlps/         # intro, history, single neuron, XOR, single hidden (5)
│   ├── week17_mlps/               # matrix notation, multilayer, multiclass, univ. approx (4)
│   ├── week18_backprop/           # training, comp graphs, backprop 1&2, hardware/software (5)
│   ├── week19_regularization/     # basic reg., early stopping, dropout & augmentation (3)
│   ├── week20_optimization/       # challenges, advanced optim, init, activations (4)
│   ├── week23_cnns_intro/         # intro, conv2d, properties, components, math, application (6)
│   ├── week24_cnns_modern/        # conv types, dilated/transposed, separable, modern CNN I&II (5)
│   ├── week25_rnns/               # intro, BPTT, modern RNN, applications (4)
│   ├── week26_attention_autoencoders/  # attention, unsupervised, manifold, AE x3 (6)
│   └── week27_vae/                # VAE (1)
├── mock_exam/
│   └── MockExamDL2026.pdf
└── lab_materials/                 # full Lab Materials folder (labs 1-10)
    ├── i2dl_intro_26.pdf
    └── lab1/ ... lab10/           # each: R (.Rmd/.pdf) + Python (.ipynb/.pdf), questions + solutions
```

Note: Week-19 lecture videos live on Google Drive and the Week-19 Ridge/Geometric-L2
readings are I2ML web chapters — links above, not downloaded as files.
