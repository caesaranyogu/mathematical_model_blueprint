
# Mathematical Modeling Journal Template

Copy this file whenever you begin a new model.

# Model: [NAME]

## 1. Phenomenon

What am I observing?

---

## 2. Question

What exactly am I trying to explain or predict?

---

## 3. Outcome

What is the dependent variable (what changes)?

Symbol:

$$
Y=
$$

Operational definition (How you measure or observe your dependent variable):

---

## 4. Candidate Variables

| Variable | Symbol | Type | How measured? |
|---|---|---|---|
| | | Input / output / state / parameter / constraint | |
| | | Input / output / state / parameter / constraint | |
| | | Input / output / state / parameter / constraint | |

---

## 5. Mechanism

What do I believe causes what?

Draw the causal chain:

$$
X_1, X_2, \ldots, X_k \xrightarrow{f(\,Z\,)} M_1, \ldots, M_j \xrightarrow{g} Y \rightleftharpoons X_1 \quad \big| \quad X \leftarrow Z \rightarrow Y \quad \big| \quad X \rightarrow C \leftarrow Y
$$

## Notation Key

| Symbol | Name | Meaning |
|---|---|---|
| $X_1, \ldots, X_k$ | Inputs | Independent variables that enter the system |
| $M_1, \ldots, M_j$ | Mediators | Intermediate variables that transmit the effect |
| $Y$ | Outcome | The thing you're predicting |
| $Z$ | Confounder / Moderator | A variable that either biases the X–Y link (confounder) or changes its strength (moderator) |
| $C$ | Collider | A variable caused by both X and Y; conditioning on it creates false associations |
| $\rightarrow$ | Direct cause | A produces B |
| $\xrightarrow{f}$ | Named mechanism | A produces B through function f |
| $\xrightarrow{f(Z)}$ | Moderation | Z alters how f transmits X's effect |
| $\rightleftharpoons$ | Feedback loop | Two variables mutually cause each other |
| $\leftarrow Z \rightarrow$ | Fork (confounder) | Z drives both sides, creating a spurious link |
| $\rightarrow C \leftarrow$ | Collider | Two causes converge on C; conditioning on C biases the relationship |
| $\big\|$ | Separator | Divides the main causal chain from structural side-patterns |


## 6. Assumptions

### Structural Assumptions
*What shape does the system take?*

- **Linearity** — Relationships between variables are linear: $Y = \beta_0 + \beta_1 X$
- **Additivity** — Effects of variables combine by addition, not interaction: $f(X_1, X_2) = f(X_1) + f(X_2)$
- **Monotonicity** — More input always means more (or always less) output; no reversals
- **Continuity** — No sudden jumps; small changes in input produce small changes in output
- **Stationarity** — The underlying process doesn't change over time
- **Symmetry** — The relationship works the same in both directions or across groups
- **Dimensionality** — Only these $k$ variables matter; all others are negligible

### Distributional Assumptions
*What does the randomness look like?*

- **Normality** — Errors or data follow a Gaussian distribution: $\epsilon \sim \mathcal{N}(0, \sigma^2)$
- **Homoscedasticity** — Variance of errors is constant across all levels of $X$
- **Independence** — Observations don't influence each other: $P(A \cap B) = P(A)P(B)$
- **Identical distribution** — Every observation is drawn from the same distribution (i.i.d.)
- **Known distribution family** — You assume the data follows a specific family (Poisson, Binomial, etc.)

### Causal Assumptions
*What causes what?*

- **No reverse causality** — $X$ causes $Y$, not the other way around
- **No omitted variable bias** — All relevant confounders $Z$ are accounted for
- **No measurement error** — $X$ is measured exactly, not a noisy proxy
- **Exogeneity** — Inputs are not correlated with the error term: $\text{Cov}(X, \epsilon) = 0$
- **No selection bias** — The sample represents the population you're modeling
- **Stable causal structure** — The causal graph doesn't rewire over time

### Boundary Assumptions
*Where does the model apply?*

- **Domain restriction** — Model is valid only for $X \in [a, b]$; extrapolation is not claimed
- **Scale invariance** — Model works at small and large scales equally
- **Temporal scope** — Model applies within a specific time window
- **Population scope** — Model applies to this group, not necessarily others
- **Steady state** — System has settled; transients have died out
- **Closed system** — No external forces act on the system during observation

### Simplification Assumptions
*What am I deliberately ignoring?*

- **Ceteris paribus** — All other variables are held constant
- **Negligible effects** — Certain variables exist but their effect is small enough to drop
- **Aggregation** — Individual differences are averaged out; the group-level model suffices
- **Deterministic approximation** — Treating a stochastic process as deterministic for tractability
- **Equilibrium** — The system is at or near a stable resting point
- **First-order approximation** — Higher-order terms (quadratic, cubic) are dropped

### Parameter Assumptions
*What do I believe about the constants?*

- **Fixed parameters** — Parameters don't change across observations or time
- **Known priors** — In Bayesian models, you assume a prior distribution: $\theta \sim \text{Prior}$
- **Identifiability** — The data can uniquely determine each parameter
- **Finite variance** — Parameters and errors have bounded variance
- **Sparsity** — Most parameters are zero or near-zero (common in high-dimensional models)

### Data Assumptions
*What do I believe about the evidence?*

- **Representative sample** — Data reflects the true population
- **Sufficient sample size** — $n$ is large enough for the method to work
- **No missing data bias** — Missing values are random, not systematic
- **Correct labeling** — Outcome variable $Y$ is accurately recorded
- **Temporal ordering** — Cause is observed before effect in the data
- **No data leakage** — Future information doesn't contaminate training data


---

## 7. Mathematical Model

Start with the simplest plausible model.

$$

$$

Explain every symbol in words.

---

## 8. Alternative Models

### Model A

$$

$$

### Model B

$$

$$

### Model C

$$

$$

---

## 9. Data Requirements

What data would allow me to estimate or test this model?

- 
- 
- 

---

## 10. Dataset Design

Rows represent:

Columns represent:

Observation frequency:

Population:

Time period:

---

## 11. Parameter Estimation

Which parameters need to be estimated?

$$

$$

How will they be estimated?

---

## 12. Validation

How will I know whether the model works?

---

## 13. Error

What metric will I use?

$$

$$

---

## 14. Sensitivity

Which assumptions matter most?

---

## 15. Causality

What alternative explanations exist?

---

## 16. Python

```python

```

---

## 17. Results

---

## 18. Failure Modes

Where could the model fail?

---

## 19. Next Experiment

What should I test next?

---

## 20. Model Version

Version:

Date:

Changes from previous version:
