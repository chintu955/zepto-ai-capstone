# Analytics Module — Findings & Written Interpretations

(Numbers below are computed directly from `sns.load_dataset('titanic')`,
which is a fixed/static dataset, so these values are reproducible by
re-running `01_eda.py`.)

## Missing Value Report & Handling

| Column | % Missing | Strategy | Justification |
|---|---|---|---|
| age | 19.87% | Median impute | Falls in the 5–30% band; median is robust to the right-skew in age. |
| embarked | 0.22% | Drop rows | Under 5%; only 2 rows affected, negligible information loss. |
| embark_town | 0.22% | Drop rows | Same 2 rows as `embarked`. |
| deck | 77.22% | Encode as "Missing" category | Too high (>30%) to impute reliably without introducing bias; kept as its own category instead of dropping the column, since cabin deck may still carry weak signal for the rows that do have it. |

Shape after cleaning: **889 rows × 15 columns** (down from 891, due to the 2 dropped `embarked` rows).

## Univariate Analysis — Outliers & Skew

- **Age outliers (IQR rule):** 65 values outside [2.5, 54.5].
- **Fare outliers (IQR rule):** 114 values outside [-26.76, 65.66] (i.e., any fare above ~65.66).
- **Fare — mean=32.10, median=14.45, mode=8.05.** Since mean > median > mode, fare is
  **right-skewed**: a small number of high-paying (mostly 1st-class) passengers pull
  the mean well above the typical (median) fare, while the single most common price
  point (mode) sits even lower, near the cheapest 3rd-class fares.

## Bivariate Analysis — Survival Rates

- **By sex:** female = 74.04%, male = 18.89%.
- **By pclass:** 1st = 62.62%, 2nd = 47.28%, 3rd = 24.24%.
- **By sex & pclass:** 1st-class women survived at 96.74%, versus 3rd-class men at
  only 13.54% — the largest gap in the dataset, showing sex and class compounded
  rather than acting independently.

## Correlation Matrix — Top 2 Strongest Pairs

Computed on `[survived, pclass, age, sibsp, parch, fare]`:

1. **pclass & fare: r = -0.548** — lower `pclass` numbers (1st class) pair with much
   higher fares, as expected since pclass is essentially a proxy for ticket price tier.
2. **sibsp & parch: r = 0.415** — passengers traveling with siblings/spouses also
   tended to travel with parents/children, i.e., families tended to book together
   rather than these being independent companions.

## Multivariate Data Story (see `charts/story_*.png`)

1. **Survival by class & sex** — women survived far more than men in every class,
   and 1st-class women survived at the highest rate overall, reflecting the
   "women and children first" protocol combined with 1st-class cabins being closer
   to lifeboats.
2. **Age distribution by survival** — survivors skew slightly younger, with a
   visible cluster of young children, consistent with children being prioritized.
3. **Fare vs age, colored by survival** — higher-fare passengers show denser
   survival markers, reinforcing that socio-economic class tracked survival
   independent of age.
4. **Survival by embarkation port** — passengers boarding at Cherbourg (C) show a
   higher survival share than Southampton (S), correlating with a higher proportion
   of 1st-class passengers embarking at C.

## Modeling — Final Comparison & Recommendation

See the console output of `02_modeling.py` for the exact run's classifier metrics
table (accuracy/precision/recall/F1/AUC for Logistic Regression, Decision Tree, and
Random Forest), the imbalance-handling comparison, GridSearchCV best parameters +
OOB score, and the regression metrics (MAE/RMSE/R²/Adjusted R²) for the fare model.

**Recommendation:** Random Forest is the recommended model to deploy. Across runs it
consistently achieves the best or near-best F1 and AUC among the three classifiers,
because it captures non-linear interactions between `sex`, `pclass`, and `fare` that
Logistic Regression cannot, while being less prone to overfitting than a single
Decision Tree. The complete fitted pipeline (preprocessing + Random Forest) is saved
as `best_pipeline.joblib` and reloads correctly on raw input (verified in
`02_modeling.py`'s final step).

## Heteroscedasticity (Regression Residuals)

The residual plot (`charts/residual_plot.png`) shows the spread of residuals
widening as predicted fare increases (a funnel shape), indicating
**heteroscedasticity** — the model's errors are not uniform across the range of
fares, and it is comparatively less precise for high-fare passengers.
