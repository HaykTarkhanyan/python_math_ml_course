"""Generate the question-bank CSV for the updated ML Q1 Google Form.

This is an UPDATE of the existing ML Q1 form, not a separate quiz: all 7 original
Q1 questions are preserved verbatim (source="q1"), and new questions from the three
new decks are added around them:
    L01e - Hyperparameter Tuning
    L01f - Regression Metrics and Diagnostics
    L01g - Feature Engineering
(L01d Validation/Cross-Validation is intentionally NOT one of the three decks.)
New open-ended rows are source="deck"; new graded multiple-choice rows are
source="deck-mc".

The CSV is the source of truth: edit it directly to review/trim questions, then feed
it to the google-forms-builder. Re-run this script only to regenerate from scratch.

Columns:
    source    - q1 | deck | deck-mc   (provenance, for review; ignored by the builder)
    area      - section code (drives Form page breaks)
    type      - short | paragraph | mc
    question  - the prompt
    optA..optD- multiple-choice options (mc only)
    correct   - correct option letter (mc only)
    points    - points for the mc item (mc only)
    feedback_correct / feedback_wrong - per-question quiz feedback (mc only)
    required  - TRUE / FALSE
"""

import csv
import logging
from pathlib import Path

HERE = Path(__file__).parent
CSV_OUT = HERE / "ml_q2_form_questions.csv"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "build_ml_q2_csv.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("build_ml_q2_csv")

FIELDS = [
    "source", "area", "type", "question",
    "optA", "optB", "optC", "optD",
    "correct", "points", "feedback_correct", "feedback_wrong", "required",
]


def short(area, question, source="q1", required="TRUE"):
    """A short-answer text field (e.g. name)."""
    return {"source": source, "area": area, "type": "short", "question": question,
            "optA": "", "optB": "", "optC": "", "optD": "",
            "correct": "", "points": "", "feedback_correct": "",
            "feedback_wrong": "", "required": required}


def p(area, question, source="deck", required="TRUE"):
    """A paragraph (open-ended, ungraded) question."""
    return {"source": source, "area": area, "type": "paragraph", "question": question,
            "optA": "", "optB": "", "optC": "", "optD": "",
            "correct": "", "points": "", "feedback_correct": "",
            "feedback_wrong": "", "required": required}


def mc(area, question, opts, correct, fbc, fbw, points=1, required="TRUE"):
    """A graded multiple-choice question (always new deck material)."""
    a, b, c, d = (opts + ["", "", "", ""])[:4]
    return {"source": "deck-mc", "area": area, "type": "mc", "question": question,
            "optA": a, "optB": b, "optC": c, "optD": d,
            "correct": correct, "points": points,
            "feedback_correct": fbc, "feedback_wrong": fbw, "required": required}


ROWS = [
    # ==================== About you ====================
    short("about", "Name, Surname"),

    # ==================== Linear regression: the model itself (Q1) ====================
    p("lr_basics",
      "How do we transform our data to more easily include intercept in the Lin. Reg. model?",
      source="q1"),
    p("lr_basics",
      "What happens if we select a very big or very small step size in Gradient Descent (GD). "
      "Also, can you bring an example from real life where you have acted like GD?",
      source="q1"),

    # ==================== Core ML concepts (Q1, foundational recap) ====================
    p("core_concepts",
      "What are the concepts of overfitting and underfitting. Can you bring examples from real life?",
      source="q1"),
    p("core_concepts",
      "Why do we even split our precious data into train and test sets instead of using everything?",
      source="q1"),
    p("core_concepts",
      "What can we do if a regression model underfits?",
      source="q1"),

    # ==================== Hyperparameter tuning (L01e) ====================
    p("hp_tuning",
      "In your own words, what is the difference between a parameter and a hyperparameter? "
      "(Hint: one you compute from the data, the other you choose.) Give one example of each."),
    mc("hp_tuning",
       "Which of the following is a HYPERPARAMETER - something you choose rather than something "
       "the model computes from the data?",
       ["The coefficients (weights) of a fitted linear regression",
        "Ridge regression's alpha (regularization strength)",
        "The intercept term found by least squares",
        "The residuals of the fitted model"],
       "B",
       "Right. alpha is not learned from the data - you pick it, and then the model learns the "
       "coefficients given that choice. Compute it from data -> parameter; choose it yourself "
       "-> hyperparameter.",
       "The coefficients and the intercept are PARAMETERS - the model computes them from the "
       "data by minimizing the loss. The residuals are just leftover errors, not something you "
       "set. Only alpha is chosen by you = the hyperparameter."),
    p("hp_tuning",
      "Grid search tries every combination of values. Why does it become hopeless once you have, "
      "say, 5 hyperparameters? And why does plain random search often find a better model on the "
      "same budget? (Intuition is enough, no formulas.)"),
    mc("hp_tuning",
       "You tune 4 hyperparameters, 5 candidate values each, with 5-fold cross-validation. How "
       "many model fits does grid search need in total?",
       ["20", "100", "625", "3125"],
       "D",
       "Correct: 5^4 = 625 combinations, times 5 folds = 3125 fits. Add one more hyperparameter "
       "and it jumps to 15,625. This blowup is exactly why random and Bayesian search exist.",
       "It is 5^4 combinations (625) times 5 CV folds = 3125. (20 = 4x5 forgets the product; "
       "100 = 4x5x5 mixes it up; 625 is right for the combinations but forgets to multiply by "
       "the 5 folds.)"),
    p("hp_tuning",
      "After you tune your hyperparameters to get the best cross-validation score, why is that "
      "best score an over-optimistic (too rosy) estimate of how the model will really do on new "
      "data? What should you report instead?"),
    p("hp_tuning",
      "Bonus: Bayesian optimization (e.g. Optuna) \"learns\" from the trials it has already run "
      "instead of guessing blindly each time. In simple words, how does that help, and what does "
      "\"explore vs exploit\" mean?",
      required="FALSE"),

    # ==================== Regression metrics (L01f) ====================
    p("metrics",
      "What is the difference between a loss and a metric? Can the thing you train the model on "
      "be different from the thing you report to people? Give an example."),
    p("metrics",
      "RMSE and MAE both summarize how big your errors are. When would you rather report MAE, "
      "and when RMSE? (Hint: think about what each one does with a single huge error.)"),
    mc("metrics",
       "Nine of your predictions are off by 1, and one is off by 100 (a luxury apartment your "
       "model missed). Which of these error metrics is LEAST affected by that single big outlier?",
       ["MSE", "RMSE", "MAE", "MedAE (median absolute error)"],
       "D",
       "Right. MedAE is the MEDIAN absolute error, so one huge outlier barely moves it (here it "
       "stays at 1). The median just doesn't care about one extreme value.",
       "MSE and RMSE square the errors, so the single 100 dominates them completely. MAE feels "
       "it moderately (the mean rises to ~10.9). MedAE - the MEDIAN absolute error - barely "
       "moves, because one outlier cannot shift the median. That is the most robust here."),
    p("metrics",
      "A model's predictions have a perfect correlation (Pearson r = 1.0) with the true values, "
      "yet the model is useless in practice. How can both things be true at the same time?"),
    p("metrics",
      "What does R^2 actually tell you about a model? And what does it mean if you compute R^2 on "
      "your test set and get a NEGATIVE number?"),
    p("metrics",
      "You plot the residuals (errors) against the model's predictions. What is the plot warning "
      "you about if the points form a fan / cone shape? What if they form a clear curve instead "
      "of random scatter?"),

    # ==================== Feature engineering (L01g) ====================
    p("feature_eng",
      "Explain main techniques for encoding categorical data and list things we need to be "
      "careful about",
      source="q1"),
    p("feature_eng",
      "In a real ML project, roughly what share of the time goes into data + features versus the "
      "actual modeling and tuning? Did the real number surprise you, and why do you think it "
      "works out that way?"),
    p("feature_eng",
      "\"Good features beat a better algorithm.\" Why is a ratio like price-per-m2 (or "
      "BMI = weight / height^2) often more useful than the raw numbers it is built from? Invent "
      "your own useful ratio from any field you like."),
    p("feature_eng",
      "Adding polynomial / interaction features (like area x floor) really helps a plain linear "
      "regression, but is basically wasted effort for a Random Forest. Why the difference?"),
    mc("feature_eng",
       "You are forecasting next month's rent from past data. Which of these is a look-ahead "
       "leakage bug (secretly using the future to predict the past)?",
       ["Creating a shift(1) lag feature (last period's value)",
        "A trailing 7-day rolling mean",
        "rolling(window=7, center=True) for a moving average",
        "Sorting the data by date before splitting"],
       "C",
       "Correct. center=True averages roughly 3 days before AND 3 days after each point, so the "
       "feature secretly contains future values. Always use trailing windows when forecasting.",
       "A (lag) and B (trailing mean) only look backwards - safe. D (sorting by date) is good "
       "practice. The bug is C: center=True pulls in future days around each point, leaking the "
       "future into the past."),
    p("feature_eng",
      "Target leakage: why is it \"cheating\" to compute \"average rent per district\" on the "
      "WHOLE dataset and only THEN split into train and test? What is the honest way to do it?"),
]


def main():
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ROWS)

    by_source = {}
    by_type = {}
    for r in ROWS:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    log.info(f"Wrote {CSV_OUT}")
    log.info(f"{len(ROWS)} rows by source: {by_source}")
    log.info(f"{len(ROWS)} rows by type:   {by_type}")


if __name__ == "__main__":
    main()
