# 11 Training LLMs — Scaling laws

> Auto-extracted from `9scale.pdf` with `pdftotext` (beamer deck). Slide titles (ALL-CAPS in the source) are marked as `##`. Inline `�` marks a glyph pdftotext could not map (often ×, a Greek letter, or a bullet); diagrams are NOT captured — see the PDF for figures.

Deep Learning for NLP Training LLMs Scaling laws
Learning goals Understand scaling laws


## Number Of Parameters: Notation
In this slide set, we use N for the number of parameters. (Unfortunately, we use P for the number of parameters in other slide sets.)


## Scaling Laws Proposed By Kaplan Et Al. (2020)
Kaplan et al. (2020)
Performance depends strongly on scale, weakly on model shape Scale means: parameters N, data D, and compute C Shape means: depth and width
Power laws Performance has power law relation with each factor N, D, C Trend spanning more than six orders of magnitude
Limits of power laws Power law for one variable only holds when not bottlenecked by the other two Performance enters regime of diminishing returns if N or D held fixed while the other increases


## Scaling Laws Proposed By Kaplan Et Al. (2020)
These power laws were huge news in 2020! Easy way to improve AI We can predict by extrapolating the early part of the training curve Power law parameters are roughly independent of model size Transfer improves with test performance
When evaluating on text with different distribution from training text, results are strongly correlated to those on the validation set Transfer to different distribution incurs a constant penalty but improves in line with performance on training set


## Power Law (Gpt3 Paper)


## Power Law (Gpt3 Paper): Questions?


## Scaling Laws: "Inefficient Convergence"
When C is fixed but N and D are not, optimal performance is achieved by training very large models and stopping significantly short of convergence That is: early stopping So the claim is: Larger language models will perform better and be more sample efficient than smaller models.


## Scaling Law For Next Word Prediction


## L(N, D)

=

1.61

+

406.4 N 0.34

+

410.7 D0.28

L(N, D) is cross entropy on new text

Source: Hoffmann et al., 2022


## Scaling Law For Next Word Prediction


## L(N, D)

=

2

+

1000 N 1/3

+

1000 D1/4

Question: Compute L(N, D) for N = 109, D = 108,

N = 1012, D = 1016,

N = 103, D = 10100 and

N = 10100, D = 104

Question: Which configuration is best?

Question: Why is it best?

Question: Which values of N and D would you choose to create an even more powerful model?

Of course, the last question is relative to the available compute budget, so try and find a reasonable answer.

COMPUTE-OPTIMAL LLMs
Given a fixed FLOP budget C, how should we trade-off model size and text size to optimize performance? Hoffmann et al., 2022
Find N and D so that FLOP(N, D) = C and L(N, D) is minimal Empirical estimates of N and D based on 400 models.
Ranging from 70 M to 16 B parameters Trained on 5 B to 400 B tokens Different results from those of Kaplan et al., 2020 Results verified using Chinchilla Chinchilla has 70 B parameters and is trained on 1.4 T tokens 4x less parameters and 4x more tokens than Gopher Chinchilla outruns Gopher and has reduced memory footprint and inference cost

COMPUTE-OPTIMAL LLMs: ARE GPT3 ETC TOO LARGE?

Source: Hoffmann et al., 2022

COMPUTE-OPTIMAL LLMs (2)
Given a fixed FLOPs budget, how should one trade off model size and the number of training tokens? We find that all three methods predict that current large models should be substantially smaller and therefore trained much longer than is currently done. Not only does Chinchilla outperform its much larger counterpart, Gopher, but its reduced model size reduces inference cost considerably and greatly facilitates downstream uses on smaller hardware.

CHINCHILLA AND THE OTHER LLMs

Source: Hoffmann et al., 2022

Source: Hoffmann et al., 2022

CHINCHILLA OUTPERFORMS OTHER LLMs: MMLU

Source: Hoffmann et al., 2022 , compared with human forecasts

CHINCHILLA OUTPERFORMS OTHER LLMs: QA

Source: Hoffmann et al., 2022


## Scaling Laws: Discussion

Question: Any doubts?


## Beyond Brute-Force Scaling Of Datasets
If you run out of data, then increase # epochs The Stack v2 already contains all public code! arxiv.org/abs/2305.16264 data quality/model size tradeoff: Training a smaller model on a smaller high-quality dataset may be better than training a huge model on a huge medium-quality dataset. For higher quality datasets, allocate more compute to model size. arxiv.org/abs/2401.02954


## Sutskever On Scaling
Ilya Sutskever: "results from scaling up pre-training . . . have plateaued." "The 2010s were the age of scaling, now we're back in the age of wonder and discovery once again. Everyone is looking for the next thing," Sutskever said. "Scaling the right thing matters more now than ever."
Reuters


## Epoch Ai Projections

Epoch AI �


## Train/Test Tradeoff (Andrew Ng)
Current models cost order of magnitude $100 million to train, and this number could reach $100 billion within a few years, according to Anthropic's Dario Amodei. Rising costs could lead companies to reallocate their gargantuan training budgets and focus on more cost-effective, application-specific approaches.


## Train/Test Tradeoff (2)
Phi4: released by Microsoft Dec 2024 14 billion parameters (small!) pretraining data: 9.8 trillion tokens (big!) pretraining data curated / synthesized


## Train/Test Tradeoff (3)

Andy L. Jones �


## Another Scaling Law: Context-Utilization Scaling
Input: Very long context, e.g., a book or several books Task: Answer a question that requires combining several pieces of evidence far apart in the context The model has to find, retain and integrate the relevant parts.
NoLiMa


## Jen-Hsun Huang On Scaling


## Main Takeaway (1)


## Main Takeaway (2)
The original scaling laws were exclusively about pretraining: What is the best model I can pretrain given constraints such as compute budget? This has shifted to a more holistic view of what we want to achieve, e.g., even if a trillion parameter model is the optimal model given pretraining constraints, a much smaller model may be preferable since it is cheaper at inference time. In the future: compound systems of many smaller models/agents rather than a single huge monolithic model? MoE is a version of that.


## What Is A Trillion?
(1:30) https://www.youtube.com/watch?v=QgUj09K7vx4
