# Deep Learning - Curated External Resources

Hand-picked, high-quality resources for the topics covered by the `lecture_i2dl` slides
(see `i2dl_slides_summary.md` and `i2dl_course_outline.md`). Organized to follow the course arc.

- Links verified via web search on **2026-06-22**. Most are **free**.
- Tag `[also in i2dl README]` = the original LMU course already links it; included here for completeness.
- Tag `[new]` = not in the LMU list; worth adding.

---

## Best starting points (if you only pick a few)

- **Books, free online:** [Understanding Deep Learning](https://udlbook.github.io/udlbook/) (Prince, 2023) and [Deep Learning: Foundations and Concepts](https://www.bishopbook.com/) (Bishop & Bishop, 2024). Both are modern, free to read, and pair theory with notebooks/figures. `[new]`
- **Video, from scratch:** [Andrej Karpathy - Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - builds backprop -> MLP -> GPT in runnable code. `[new]`
- **Visual intuition:** [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZZWrBYkx7Otcjr3eCLZDCgfpqnxMY29s). `[new]`
- **Interactive:** [TensorFlow Playground](https://playground.tensorflow.org/) and [CNN Explainer](https://poloclub.github.io/cnn-explainer/).

---

## 1. Foundational books (all free to read online)

| Book | Link | Notes |
|---|---|---|
| Prince - *Understanding Deep Learning* (2023) | https://udlbook.github.io/udlbook/ | Free PDF + Python notebooks + slides. Modern, concise, well-illustrated. `[new]` |
| Bishop & Bishop - *Deep Learning: Foundations and Concepts* (2024) | https://www.bishopbook.com/ | Free digital version, exercise solutions, downloadable figures. Strong on fundamentals. `[new]` |
| Goodfellow, Bengio, Courville - *Deep Learning* (2016) | https://www.deeplearningbook.org/ | The classic reference. Math-heavy. `[also in i2dl README]` |
| Zhang et al. - *Dive into Deep Learning* (d2l.ai) | https://d2l.ai/ | Interactive book with PyTorch/TF/JAX/MXNet code for every concept. `[also in i2dl README]` |
| Nielsen - *Neural Networks and Deep Learning* | http://neuralnetworksanddeeplearning.com/ | Beginner-friendly; best free intro to backprop via digit recognition. `[new]` |

## 2. Video courses and lecture series

| Course | Link | Covers |
|---|---|---|
| Karpathy - Neural Networks: Zero to Hero | https://karpathy.ai/zero-to-hero.html (code: https://github.com/karpathy/nn-zero-to-hero) | Backprop, micrograd, makemore, batchnorm, building a GPT. `[new]` |
| 3Blue1Brown - Neural Networks | https://www.youtube.com/playlist?list=PLZZWrBYkx7Otcjr3eCLZDCgfpqnxMY29s | Visual intuition for NNs, gradient descent, backprop, attention. `[new]` |
| fast.ai - Practical Deep Learning for Coders | https://course.fast.ai/ | Hands-on, top-down practical DL. `[also in i2dl README]` |
| Stanford CS231n - DL for Computer Vision | https://cs231n.stanford.edu/ (notes: https://cs231n.github.io/) | CNNs, training, modern vision, transformers/diffusion. `[also in i2dl README]` |
| Stanford CS230 - Deep Learning (Ng) | https://cs230.stanford.edu/ | CNNs, RNNs, Adam, dropout, batchnorm, init - maps closely to this course. `[new]` |
| Stanford CS224n - NLP with Deep Learning | https://web.stanford.edu/class/cs224n/ | Word vectors, RNN/LSTM, seq2seq, attention, Transformers, LLMs. `[new]` |
| Stanford CS236 - Deep Generative Models | https://deepgenerativemodels.github.io/ (videos: https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8) | VAEs, GANs, autoregressive, flows, diffusion, score-based. `[new]` |

---

## 3. Topic-by-topic (mapped to the i2dl folders)

### Intro / history (`intro`)
- Nielsen book ch.1 - what neural nets are, from scratch: http://neuralnetworksanddeeplearning.com/chap1.html `[new]`
- 3Blue1Brown ch.1 "But what is a neural network?": https://www.youtube.com/watch?v=aircAruvnKk `[new]`

### From a neuron to MLPs, universal approximation (`mlps`)
- 3Blue1Brown full NN series (see above). `[new]`
- "The Matrix Calculus You Need For Deep Learning" (Parr & Howard): https://explained.ai/matrix-calculus/ `[also in i2dl README]`
- Universal approximation, visual proof (Nielsen ch.4): http://neuralnetworksanddeeplearning.com/chap4.html `[new]`

### Backprop and computational graphs (`opt1`)
- Karpathy - "Yes you should understand backprop": https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b `[also in i2dl README]`
- Karpathy - micrograd lecture (builds autograd by hand): https://www.youtube.com/watch?v=VMj-3S1tku0 `[new]`
- Nielsen ch.2 - how backprop works: http://neuralnetworksanddeeplearning.com/chap2.html `[new]`

### Optimization: SGD, momentum, Adam, init, batchnorm (`opt2`)
- Ruder - "An overview of gradient descent optimization algorithms": https://www.ruder.io/optimizing-gradient-descent/ (paper: https://arxiv.org/abs/1609.04747) `[also in i2dl README]`
- Distill - "Why Momentum Really Works": https://distill.pub/2017/momentum/ `[also in i2dl README]`
- Karpathy - "A Recipe for Training Neural Networks": https://karpathy.github.io/2019/04/25/recipe/ `[also in i2dl README]`
- Adam paper (Kingma & Ba, 2014): https://arxiv.org/abs/1412.6980 `[new]`
- Batch Normalization paper (Ioffe & Szegedy, 2015): https://arxiv.org/abs/1502.03167 `[new]`

### Regularization: dropout, weight decay, early stopping, augmentation (`regu`)
- Dropout paper (Srivastava et al., 2014): https://jmlr.org/papers/v15/srivastava14a.html `[new]`
- Goodfellow book ch.7 "Regularization for Deep Learning": https://www.deeplearningbook.org/contents/regularization.html `[new]`
- "Regularization for Deep Learning: A Taxonomy": https://arxiv.org/abs/1710.10686 `[also in i2dl README]`

### CNNs: convolution, pooling, architectures, applications (`cnn1`, `cnn2`, `cnn3`)
- CNN Explainer (interactive, Polo Club): https://poloclub.github.io/cnn-explainer/ `[new]`
- CS231n CNN notes: https://cs231n.github.io/convolutional-networks/ `[also in i2dl README]`
- Distill - "Computing Receptive Fields of CNNs": https://distill.pub/2019/computing-receptive-fields/ `[also in i2dl README]`
- Distill - "Deconvolution and Checkerboard Artifacts" (transposed conv): https://distill.pub/2016/deconv-checkerboard/ `[also in i2dl README]`
- "A guide to convolution arithmetic for deep learning" (Dumoulin & Visin): https://arxiv.org/abs/1603.07285 `[new]`
- Key architecture papers: AlexNet https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html, VGG https://arxiv.org/abs/1409.1556, ResNet https://arxiv.org/abs/1512.03385, U-Net https://arxiv.org/abs/1505.04597 `[new]`

### RNNs, LSTM, GRU (`rnn`)
- colah - "Understanding LSTM Networks": https://colah.github.io/posts/2015-08-Understanding-LSTMs/ `[also in i2dl README]`
- Karpathy - "The Unreasonable Effectiveness of Recurrent Neural Networks": https://karpathy.github.io/2015/05/21/rnn-effectiveness/ `[new]`
- d2l.ai RNN chapter: https://d2l.ai/chapter_recurrent-neural-networks/ `[new]`

### Attention and Transformers (`rnn/slides-attention`)
- Jay Alammar - "The Illustrated Transformer": https://jalammar.github.io/illustrated-transformer/ `[new]`
- 3Blue1Brown - "Attention in transformers, step by step": https://www.3blue1brown.com/lessons/attention/ `[new]`
- The Annotated Transformer (Harvard NLP, line-by-line PyTorch): https://nlp.seas.harvard.edu/annotated-transformer/ `[new]`
- Karpathy - "Let's build GPT from scratch": https://www.youtube.com/watch?v=kCc8FmEb1nY `[new]`
- "Attention Is All You Need" (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762 `[new]`

### Autoencoders and VAEs (`ae`, `genmod`)
- Lilian Weng - "From Autoencoder to Beta-VAE": https://lilianweng.github.io/posts/2018-08-12-vae/ `[new]`
- Jaan Altosaar - "What is a variational autoencoder?": https://jaan.io/what-is-variational-autoencoder-vae-tutorial/ `[new]`
- Doersch - "Tutorial on Variational Autoencoders": https://arxiv.org/abs/1606.05908 `[also in i2dl README]`
- CS236 lecture notes: https://deepgenerativemodels.github.io/notes/ `[new]`

### GANs (`gan1`)
- Lilian Weng - "From GAN to WGAN": https://lilianweng.github.io/posts/2017-08-20-gan/ `[new]`
- GAN paper (Goodfellow et al., 2014): https://arxiv.org/abs/1406.2661 `[new]`
- Goodfellow - NIPS 2016 GAN Tutorial: https://arxiv.org/abs/1701.00160 `[new]`
- Conditional GANs (Mirza & Osindero, 2014): https://arxiv.org/abs/1411.1784 `[new]`

### Adversarial examples and training (`adver`)
- "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., FGSM, 2014): https://arxiv.org/abs/1412.6572 `[new]`
- "Towards Deep Learning Models Resistant to Adversarial Attacks" (Madry et al., PGD, 2017): https://arxiv.org/abs/1706.06083 `[new]`
- CleverHans library (attacks/defenses benchmark, JAX/PyTorch/TF2): https://github.com/cleverhans-lab/cleverhans `[new]`
- CS231n lecture - Adversarial Examples and Adversarial Training: https://www.youtube.com/watch?v=CIfsB_EYsVI `[new]`

---

## 4. Interactive tools and visualizations

- TensorFlow Playground - train a tiny net in the browser: https://playground.tensorflow.org/ `[also in i2dl README]`
- CNN Explainer - step through a live CNN: https://poloclub.github.io/cnn-explainer/ `[new]`
- ConvNetJS (Karpathy) - neural nets in the browser: https://cs.stanford.edu/people/karpathy/convnetjs/ `[new]`
- Netron - view/inspect saved model architectures: https://netron.app/ `[new]`
- Distill.pub - in-depth interactive explanations: https://distill.pub/ `[also in i2dl README]`

## 5. Blogs worth following

- Lil'Log (Lilian Weng) - https://lilianweng.github.io/ - deep, careful surveys (VAE, GAN, attention, diffusion). `[new]`
- colah's blog (Chris Olah) - https://colah.github.io/ - intuition-first explanations. `[also in i2dl README]`
- Jay Alammar - https://jalammar.github.io/ - illustrated guides to transformers/LLMs. `[new]`
- Sebastian Ruder - https://www.ruder.io/ - optimization and NLP. `[new]`
- Andrej Karpathy - https://karpathy.github.io/ - backprop, RNNs, training recipes. `[also in i2dl README]`

---

*Generated with web-search-verified links on 2026-06-22. If a link rots, search the title - these are all well-known, stable resources.*
