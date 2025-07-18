---
title: "Python, Math, Machine Learning Course" 
author: "[Metric.am](https://metric.am/)"
---

I've just renamed a `qmd` file to `md`, that's why the file is not rendering properly.

{{< include misc/links.qmd >}}

# 🐍 Python

## 01 - Intro
::: {.callout-note}
[📚 Ամբողջական նյութը](python/01_intro.ipynb)

Առաջին ծանոթություն Python-ի հետ։ 1) Ինչպես արժեքներ տպել (print (sep, end)), 2) մեկնաբանութուններ (comments) ավելացնել, 3) փոփոխականներ ստեղծել, 4) թվեր և թվաբանական գործողություններ կատարել, 5) մուտք (input) ստանալ։ 

Նաև ծանոթանում ենք տնային անելու հարթակի՝ [Profound](https://profound.academy) հետ:

Գործնական դասի ժամանակ էլ արել ենք հետևյալ խնդիրները՝

1. Կիսամյակային գնահատականի հաշվիչ
2. Քառակուսի հավասարման արմանտների գտնում
3. հեշտ էր չէ՞ (2.1)
4. ուսանողներ և խնձորներ 2 (4.3)
5. թվաբանական պրոգրեսիա (4.1)

Դասերի վերաբերյալ կարող եք անանուն հայտնել Ձեր կարծիքը [այստեղ](https://forms.gle/K616aM5cpXsnJmbAA)
:::
### 📺 Տեսանյութեր
1. [Տեսադասը](https://youtu.be/_C3sP0X7U_E)
2. [Profound-ից օգտվելու վիդեո](https://youtu.be/BaQ0-hrcRtI)
3. [Գործնական դաս](https://youtube.com/watch?v=Xdb7ivwWqxE)


2023 (կարելի ա բաց թողել)

1. [Դասախոսությունը](https://youtube.com/watch?v=M5ur3GqsLh0) (մեծամասամբ իմաստ չկա նայելու)
2. [Գործնականը](https://youtube.com/watch?v=Fwk1PyLotOg) (մեծամասամբ իմաստ չկա նայելու)

### 🏡Տնային
1. Profound [բաժին 1](https://profound.academy/hy/python-introduction/t-python-qWPtjbycGqzmt41dkWyP) (Մուտք և ելք) - լրիվ
2. Profound [բաժին 3](https://profound.academy/hy/python-introduction/t-python-qWPtjbycGqzmt41dkWyP) (Փոփոխականներ և ամբողջ թվեր) - լրիվ
3. Ոչ պարտադիր - կարող եք անանուն հայտնել Ձեր կարծիքը դասի վերաբերյալ [այստեղ](https://forms.gle/K616aM5cpXsnJmbAA)

```{=html}
<details>
    <summary>Նշումներ</summary>
    <ul>
        <li>Տնայինը անելիս մոտեցեք են սկզբունքով որ x խնդիրը լուծելիս դուք մենակ գիտեք են ամենինչը ինչ profound-ը մինչև x-ին հասնելը ներկայացրելա ա։ Երբեմն օգտագործող գործիքները սահմանափակելու դեպքում ա խնդիրը իսկականից օգուտ տալիս։ Օրինակի համար եթե խնդիր լինի որտեղ պետք ա սորտավորել թվերը ու դուք գիտեք <code>sort</code> հրամանի մասին՝ է հա, կարաք օգտագործեք էդ հրամանը, խնդիրը լուծվի՝ բայց արդյունքում նորմալ չեք սովորի էլի։</li>
        <li>Եթե քիչ ժամանակ ունեք կարաք բաց թողեք Բաժին 1 - 5, 6, 14 խնդիրները</li>
        <li>Բաժին 3 13-ում պետք ա զուտ ցանկացած թիվ տպեք</li>
        <li>Բաժին 3-ի վերջին երկու խնդիրները լիքը բզբզալու են, խորհուրդ կտանք շատ ժամանակ տրամադրեք իրենց</li>
        <li>Profound-ը ներկայացնող <a href="https://youtu.be/BaQ0-hrcRtI">վիդեո</a></li>
        <li>Շուտ եմ ասել GPT ու նման գործիքներից օգտվել չկա</li>
        <li>Եթե հարցեր լինի՝ անպայման խաբար արեք (կառալյոկի պահը հիշեք)</li>
    </ul>
</details>
```

## 2 - Conditions / Պայմաններ
::: {.callout-note}
[📚 Ամբողջական նյութը](python/02_conditions.ipynb)

Սովորում ենք ինչպես աշխատացնել տարբեր ծրագրեր կախված որոշ պայմաններից։ Նախ ծանոթանում ենք `boolean` տվյալների տեսակին (True, False արժեքները ընդունող), ապա որոշ համեմատության գործողություններին (==, >, is, ...), հետո ծանոթանում ենք `if` (եթե) բլոկին ու իրա հետ եկող `elif`, `else` կտորներին։ Վերջում նայում ենք ներդրված պայմանները (if-ի մեջ if) ու համեմատաբար նոր գրելաձև `match`-ով։ 

Գործնական դասերին (հիմիկվա, ու 2023-ի) նայում ենք լրիվ տարբեր խնդիրներ, նենց որ կարող եք երկու դասն էլ նայել։ Նաև հասանելի ա տնայինների քննարկման տեսագրություն 2023-ից։
:::
### 📺 Տեսանյութեր
1. [Տեսադասը](https://youtube.com/watch?v=PsTB0hj95OM)
2. [Գործնական դաս](https://youtube.com/watch?v=2GMufITVgt4)


2023

1. [Դասախոսությունը](https://youtube.com/watch?v=UZ6lVl0kOlo) (նյութը նույնն ա, դասը անցկացնողները տարբեր)
2. [Գործնականը](https://youtube.com/watch?v=wx3wn5SPt3g) (խնդիրները լրիվ տարբեր են)
3. [Տնայինների քննարկում](https://youtube.com/watch?v=6Aktg75ZquA)

### 🏡Տնային
1. Profound [բաժին 5](https://profound.academy/hy/python-introduction/b-vE3xZikKFjcgaFdTsGvg) (Պայմաններ) - լրիվ
2. Profound [բաժին 7](https://profound.academy/hy/python-introduction/n-if-8uPJetpgIBlFPXCTxvEp) (Ներդրված պայմաններ) - լրիվ
3. Profound [բաժին 9](https://profound.academy/hy/python-introduction/flszat-nMh1l21kkviihZwWp6kf) (Փոփոխականներ և տիպեր) - 1 - 10 (ներառյալ)

```{=html}
<details>
    <summary>Նշումներ</summary>
        <ul>
            <li>Բաժին 5-ը գրելիս կարելի ա օգտվել մենակ <code>if</code>-ից ու <code>else</code>-ից (<code>elif</code> ենթադրեք որ չգիտեք)</li>
            <li><a href="https://profound.academy/hy/python-introduction/d-tNWtCx4NdIhu2lDVRR6I">Դժվար պայմաններ</a> (7.14)-ը լուծելիս եթե շատ երկար լինի Ձեր լուծումը, մի հատ էլ հետո մտածեք ոնց կարաք ավելի կարճ ու կոկիկ դարձնեք bool փոփոխականներ սահմանելով</li>
            <li>Եթե կուզեք կարաք Բաժին 9-ը փորձեք մինչև վերջ անել՝ հաջորդ տնայինում դա լինելու ա</li>
        </ul>
</details>
```

## 3 - String, list, range, functions on floats/lists
Done, add links later

## **4 - Loops / Ցիկլեր**
Done, add links later

## 5 - **List/String Methods + Ternary Operators, List Comprehensions**
Done, add links later

## 6 - **Tuple, Set, Dictionary**
Done, add links later

## 7, 8 - Functions (գուցե երկու շաբաթ հատկացնենք)
Done, add links later

## 9 - Terminal, Working with multiple files, file I/O, Packages (os, random, time, tqdm)
Done, add links later
## 10 - Git / GitHub, Venvs, Anaconda + PEP8, clean code/architecture
Done, add links later

## 11 - Exception Handling
Done, add links later

## 12 - Streamlit, Recustions, leftover material
Done, add links later

## 13 - Decorators
Done, add links later

## 14 OOP 1: Classes
Done, add links later

## 15 OOP 2: Inheritance, Polymorphism
Done, add links later

## 16 OOP 3: Encapsulation, Abstraction
Done, add links later

## 17 Data Classes, Generators, Iterators, Context Managers
Done, add links later

## 18 Final Project: YouTube + Translator
Done, add links later

# 📦Packages
## 01 OpenAI (timestamp generator project) [19]
Done, add links later

## Data Science Packages
02 July, Wednesday
[] NumPy

04 July, Friday
[] Pandas 1 

06 July, Sunday 
[] Pandas 2 + Practical

09 July, Wednesday
[] Data Visualization

11 July, Friday
[] Project


July 2, Wednesday
[] Pandas 1 

July 4, Friday
[] Pandas 2 + Profiling

July 6, Sunday (perhaps skip)
[] Data Visualization

July 9, Wednesday
[] Project 1 

July 11, Friday
[] Project 2

July 14 - 21 - Break

## General Packages
[] Logging, Unittest (Pytest), Argparse (other CLI)

[] pydantic
[] python-dotenv
[] Flask / FastAPI 

[] Scraping 

[] Path
[] maybe a bit of manim just for fun
[] Collections / functools
[] yaml
[] icecream
[] Make
[] docker
[] pytest 
[] smthing argparse like
[] dvc
[] packaging
[] zip
[] smtp
[] numba 
[] Sweetviz / pandas profiling

## 15 - Logging, Unittest (Pytest), Argparser

## 16 - Scraping 

## 17 - Flask / FastAPI

## 18 - NumPy 

## 19-20 - Pandas

## 21-22 - Data Visualization

## 23 - Some other packages (Streamlit, Dask, Sweetviz, Numba, …)

# 📈 Math

## 🧮 20-22.5 Linear Algebra

- Vectors, vector operations, dot product, norm
- Vector spaces and subspaces
- Matrices, matrix operations
- Geometric interpretation of matrices
- Row echelon form
- Determinant in 2x2 and 3x3 cases, trace
- Determinant in general case
- Systems of linear equations
- Gauss-Jordan elimination
- Inverse matrix
- Linear independence
- Basis, rank, dimension
- Eigenvalues and eigenvectors
- Positive/negative definite matrices
- Decompositions

## 📈 22.5 - 24 Calculus

- Limit of sequence and function
- Derivative
- Extrema of a function
- Taylor polynomials
- Indefinite integral, definite integral
- Partial derivative
- Gradient, directional gradient
- More topics

## ⛰️ 25 - 27 Optimization

- Quadratic forms and Sylvester’s criterion
- Gradient Descent
- Momentum
- AdaGrad / RMSProp / ADAM
- Second order methods
- Constrained optimiziation
- Evolutionary algorithms
- Bayesian optimization
- Multicriteria optimization

## 🎲 28 - 29 Probability Theory

- Sample space, events, probability
- Independence
- Conditional probability, total probability
- Bayes rule
- Geometric probability
- Random variable
- PMF, CDF, PDF
- Expected value, variance
- Covariance and correlation
- Distributions
- Laws of large numbers
- Central limit theorem

## 📊30 - 31 Statistics

- Point estimation: Mean, median, mode
- Estimator properties
- MAP / MLE
- Confidence intervals and hypothesis testing
- P-values, type I and type II errors

# 🤖 Machine Learning

## 32 Linear Regression

- Assumptions
- Loss
- Gradient based optimization
- Normal Equation
- Interpretation of Coefficients

## 33 - 34 Main Concepts

- Encoding categoricals
- Feature scaling
- Train Val Test split (data leakage issue)
- (Stratified) Cross validation
- Regression evaluation metrics

## 35 - 36 More Regression + Main Concepts 2

- Polynomial Regression
- Under / Overfitting
- Regularization
    - Ridge
    - Lasso
- Hyperparameter Search
- Feature Engineering
- Outliers
- Threshold tuning

## 37 Logistic Regression

- Logistic regression
- Log odds
- Classification evaluation metrics

## 38 Trees

- Decision tree
- Bagging
- Boosting
- Notable models (i.e. LightGBM)

## 39 Model interpretation and Feature selection

## 40 Unsupervised Learning

- KMeans
- DBSCAN
- Hierarchical
- Clustering evaluation metrics

## 41 - 42 Neural Networks

## 43 - 44 Intro to Computer Vision

## 44 - 45 Intro to Natural Language Processing

## 46 - 47 Intro to Gen AI

## Գուցե նաև KNN, SVM, Information Theory, Gaussian Process

## Final Project

# quarto html block 

```{=html}
<a href="http://s01.flagcounter.com/more/1oO"><img src="https://s01.flagcounter.com/count2/1oO/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_0/pageviews_1/flags_0/percent_0/" alt="Flag Counter"></a>
```