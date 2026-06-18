/** AUTO-GENERATED Apps Script for the ML Q1 review form.
 * Paste into a new Apps Script project (script.new) and run buildForm().
 * No Drive / Sheets scope needed - questions are embedded below.
 */
var FORM_TITLE = "ML Q1 review - L01 / L01b / L01c";
var FORM_DESCRIPTION = "Open-ended review questions covering the first three slidesets: L01 (intro to linear regression), L01b (derivations), L01c (preprocessing). After you submit, click 'View score' to see the reference answers for each question.";
var ROWS = [
  {
    "type": "short",
    "title": "Name, Surname (you can also stay anonymous if you'd like)",
    "required": false,
    "area": "about"
  },
  {
    "type": "paragraph",
    "title": "If you'd like to share feedback regarding the lessons - please do it here. The more negative the feedback, the more helpful it will be. Thanks!",
    "required": false,
    "area": "feedback"
  },
  {
    "type": "paragraph",
    "title": "Եթե ինչ-որ անիկդոտ պատմեք պատասխանները կարդալուց ավելի շատ կուրախանամ։ Բայց դե պարտադիր չի իհարկե։ :^)",
    "required": false,
    "area": "feedback"
  },
  {
    "type": "paragraph",
    "title": "1. What are the main types of ML (supervised, unsupervised, reinforcement) and how do they differ",
    "required": true,
    "area": "l01_intro",
    "feedback": "Supervised learning uses labeled data (each example has an input x and a known target y) and learns a mapping x -> y; covers regression and classification. Unsupervised learning has no labels and finds structure in the data: clustering, dimensionality reduction, density estimation. Reinforcement learning has an agent interacting with an environment, receiving rewards, and learning a policy that maximizes long-term reward (Go, robotics, game-playing)."
  },
  {
    "type": "paragraph",
    "title": "2. Explain the idea of the linear regression learning loop - what are its main ingredients",
    "required": true,
    "area": "l01_intro",
    "feedback": "Four ingredients. (1) Model / hypothesis space: for linear regression, all functions f(x) = theta^T x. (2) Loss: how wrong a single prediction is - for OLS, squared error (y - f(x))^2. (3) Optimization: a way to find the parameters that minimize the empirical risk (sum of losses) - either the closed-form normal equation or iterative gradient descent. (4) Evaluation: check performance on held-out data, otherwise we just measure how well we fit the training set."
  },
  {
    "type": "paragraph",
    "title": "3. How can we derive a formula for the optimal parameters that minimize Mean Squared Error",
    "required": true,
    "area": "l01b_derivations",
    "feedback": "Write the empirical risk in matrix form: R(theta) = ||X theta - y||^2 = (X theta - y)^T (X theta - y). Expand: theta^T X^T X theta - 2 theta^T X^T y + y^T y. Take the gradient w.r.t. theta using the rules d/dtheta (theta^T A theta) = 2 A theta and d/dtheta (a^T theta) = a: gradient = 2 X^T X theta - 2 X^T y. Set to zero: X^T X theta = X^T y. Solve: theta_hat = (X^T X)^{-1} X^T y. This is the normal equation - unique global minimum when X has full column rank."
  },
  {
    "type": "paragraph",
    "title": "4. How can we easily include an intercept in our linear regression model",
    "required": true,
    "area": "l01b_derivations",
    "feedback": "Prepend a column of 1s to the design matrix X so each row becomes (1, x_1, ..., x_p). Equivalently, pad the feature vector x with a leading 1. Then theta = (theta_0, theta_1, ..., theta_p) and the product x^T theta automatically equals theta_0 + theta_1 x_1 + ... + theta_p x_p - the intercept is absorbed as just another coefficient. This means we do not carry theta_0 separately in the math or the code; one matrix multiplication handles everything, and the same normal equation works without modification."
  },
  {
    "type": "paragraph",
    "title": "5. What should we change (and how) in linear regression so we can model polynomial relationships",
    "required": true,
    "area": "l01b_derivations",
    "feedback": "Do not change the model - change the features. Apply a basis expansion: replace x with (1, x, x^2, x^3, ..., x^d), and for multivariate features add cross terms like x_1 x_2. Then run ordinary linear regression on the expanded design matrix. The model is still LINEAR IN THE PARAMETERS theta (which is what 'linear regression' really means), so the same normal equation applies - only the columns of X change. Choosing d too high overfits, too low underfits (U-shape of test error)."
  },
  {
    "type": "paragraph",
    "title": "6. What hyperparameter do we pick when running gradient descent, and how should we choose its value",
    "required": true,
    "area": "l01_intro",
    "feedback": "The learning rate (step size) alpha. Too small -> painfully slow convergence; too large -> overshooting, oscillation, or divergence. How to choose: (1) try values on a log scale (1, 0.1, 0.01, 0.001, ...) and plot the loss curve - keep the largest one that still decreases monotonically; (2) use a learning rate schedule that decays over iterations (e.g. 1/sqrt(t)); (3) use adaptive optimizers (Adam, RMSProp) that adjust the step per parameter. Other related hyperparams: batch size, number of epochs, momentum."
  },
  {
    "type": "paragraph",
    "title": "7. Why is vectorization (e.g. X @ theta) faster than a Python for loop",
    "required": true,
    "area": "l01b_derivations",
    "feedback": "Three reasons. (1) BLAS / LAPACK: numpy calls into optimized C/Fortran linear algebra libraries that use SIMD instructions and exploit CPU cache - a single matrix multiply runs on highly tuned native code. (2) No interpreter overhead: a Python for loop pays bytecode dispatch, attribute lookups, and number boxing per iteration; vectorized ops do the loop in C. (3) Memory layout: contiguous numpy arrays are cache-friendly; Python list-of-floats wastes pointer indirection. For n = 10^6 rows the vectorized version is often 100-1000x faster."
  },
  {
    "type": "paragraph",
    "title": "8. Why wouldn't we always use the normal equation - when does gradient descent become a better choice",
    "required": true,
    "area": "l01b_derivations",
    "feedback": "The normal equation requires computing (X^T X)^{-1}, which is O(p^3) and needs the matrix to fit in memory. Gradient descent is preferred when: (a) p is very large (each GD step is only O(np)); (b) data does not fit in memory and we need streaming / mini-batch updates; (c) X^T X is singular or ill-conditioned; (d) the loss has no closed form (most non-OLS models: logistic regression, neural nets, any non-convex setup) - in those cases iterative optimization is the only option."
  },
  {
    "type": "paragraph",
    "title": "9. How can we handle missing data (NaN values) - name at least 2 strategies",
    "required": true,
    "area": "l01c_preprocessing",
    "feedback": "(1) Drop rows or columns with NaN - simple but discards information; OK only when missingness is rare and random (MCAR). (2) Simple imputation - fill with the column mean / median / mode; shrinks variance and biases downstream models. (3) Missing-indicator trick - keep the imputed value AND add a binary 'is_missing' column so the model can learn that 'missing' itself carries signal. (4) Model-based imputation (KNNImputer, IterativeImputer) - predict the missing value from the other features; more accurate but heavier and can leak if not done carefully."
  },
  {
    "type": "paragraph",
    "title": "10. What categorical encoding techniques do you know (one-hot, label/ordinal, target etc.) and what is a downside of each",
    "required": true,
    "area": "l01c_preprocessing",
    "feedback": "One-hot: one binary column per level. Downside: explodes the column count for high-cardinality features and creates perfect multicollinearity with an intercept (dummy trap). Label / ordinal: assign integers 0, 1, 2, ... to levels. Downside: imposes a false numeric ordering on unordered categories - safe ONLY for truly ordered features. Target encoding: replace each level with the mean of y for that level. Downside: data leakage if fit on the full dataset; needs out-of-fold computation and smoothing. Frequency encoding: replace level with its count. Downside: collisions between levels with similar frequency."
  },
  {
    "type": "paragraph",
    "title": "11. What 2 things should we be careful about when applying one-hot encoding",
    "required": true,
    "area": "l01c_preprocessing",
    "feedback": "(1) The dummy trap: with k one-hot columns AND an intercept, the columns sum to 1 across rows and are linearly dependent - X^T X becomes singular and the normal equation breaks. Fix: drop one column (drop_first=True), or drop the intercept. Tree models do not care. (2) High cardinality: a column with 1000 unique values explodes into 1000 columns, blowing up memory and risking severe overfitting. Fix: group rare levels into 'Other', or switch to target / frequency encoding for the high-cardinality column."
  },
  {
    "type": "paragraph",
    "title": "12. Why would we want to scale our features, and what 2-3 scalers can we use to do it",
    "required": true,
    "area": "l01c_preprocessing",
    "feedback": "Scaling matters for: gradient-descent-based models (badly-scaled features stretch the loss surface and slow convergence), distance-based models (KNN, k-means, SVM with RBF - the largest-scale feature dominates the distance), and regularized linear models (L1 / L2 penalty depends on coefficient magnitude). Scalers: (1) StandardScaler = (x - mean) / std, assumes roughly Gaussian. (2) MinMaxScaler = (x - min) / (max - min), maps to [0, 1] but is sensitive to outliers. (3) RobustScaler = (x - median) / IQR, robust to outliers - prefer when the column has heavy tails."
  },
  {
    "type": "paragraph",
    "title": "13. Why is label encoding dangerous for unordered categories - give an example",
    "required": true,
    "area": "l01c_preprocessing",
    "feedback": "Label encoding maps categories to integers (0, 1, 2, ...). For unordered categories like {Yerevan, Gyumri, Vanadzor} -> {0, 1, 2}, the encoding falsely tells a linear or distance-based model that 'Vanadzor is twice Gyumri' and 'Yerevan is closer to Gyumri than to Vanadzor'. The model will use this fake ordering to make decisions and predict garbage. Example: encoding cheese types {brie=0, cheddar=1, gouda=2} in a price-prediction model - the model assumes gouda is 'more' of something than cheddar. Use one-hot or target encoding instead. Label encoding is safe ONLY for truly ordered features like {small, medium, large} or {bad, ok, good}."
  },
  {
    "type": "paragraph",
    "title": "14. What could go wrong with target encoding - how do we avoid it",
    "required": true,
    "area": "l01c_preprocessing",
    "feedback": "Target encoding replaces each category with the mean of the target for that category - so the feature literally contains information about y. The risk is data leakage: if we compute the encoding on the full training set and then train the model on the same rows, the model gets to peek at y through the feature, inflating training scores and overfitting. Fixes: (1) out-of-fold target encoding - for each row, compute the encoding using OTHER rows only (k-fold style), so the row's own y never enters its own feature value; (2) smoothing - shrink each per-category mean toward the global mean so rare categories do not memorize their target; (3) always fit the encoder on the training split and apply to test - never on the combined data. Same rule that applies to scalers and imputers."
  },
  {
    "type": "paragraph",
    "title": "15. How do log or power transforms help with skewed features",
    "required": true,
    "area": "l01c_preprocessing",
    "feedback": "Right-skewed features (income, house price, web traffic) have long heavy tails - a few huge values dwarf the rest. Linear models assume the feature has a roughly linear effect on y, so heavy tails let extreme values drive the fitted coefficients. A log transform (log(1+x) for non-negative x) compresses large values and stretches small ones, pulling the distribution closer to symmetric. Box-Cox is a parameterized power transform that finds the best exponent for strictly positive features; Yeo-Johnson is its generalization that works for negative values too. Effects: more symmetric residuals, better-behaved coefficients, often better predictive performance for linear and distance-based models. Tree models do not need this."
  }
];

function buildForm() {
  var form = FormApp.create(FORM_TITLE);
  form.setDescription(FORM_DESCRIPTION);
  form.setIsQuiz(true);
  form.setCollectEmail(false);
  for (var i = 0; i < ROWS.length; i++) {
    var r = ROWS[i];
    if (r.section_before) {
      form.addPageBreakItem().setTitle(r.section_before);
    }
    var item;
    if (r.type === 'short') {
      item = form.addTextItem();
    } else {
      item = form.addParagraphTextItem();
    }
    item.setTitle(r.title);
    item.setRequired(!!r.required);
    if (r.feedback) {
      item.setPoints(1);
      item.setGeneralFeedback(FormApp.createFeedback().setText(r.feedback).build());
    }
  }
  Logger.log('Built ' + ROWS.length + ' items.');
  Logger.log('Edit URL: ' + form.getEditUrl());
  Logger.log('Published URL: ' + form.getPublishedUrl());
}
