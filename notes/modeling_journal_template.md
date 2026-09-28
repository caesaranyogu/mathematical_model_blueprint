
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


# Mathematical Model Types

A reference for Section 7 and Section 8 of the Modeling Journal.

---

## 1. Linear Model

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_k X_k + \epsilon
$$

The workhorse. Assumes the output is a weighted sum of inputs plus noise. Used when relationships are approximately proportional — predicting house prices from square footage, salary from years of experience. Start here, upgrade only when it fails.

---

## 2. Polynomial Model

$$
Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \cdots + \beta_n X^n + \epsilon
$$

A linear model in disguise — linear in the coefficients, nonlinear in the input. Captures curves, peaks, and valleys. Used when the relationship bends — diminishing returns, U-shaped cost curves, projectile trajectories.

---

## 3. Logistic Model

$$
P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}
$$

Maps any input to a probability between 0 and 1 via the sigmoid function. Used for binary outcomes — will the customer churn, is the email spam, does the patient have the disease.

---

## 4. Exponential Growth / Decay

$$
Y(t) = Y_0 \, e^{rt}
$$

Output grows (or shrinks) at a rate proportional to its current size. $r > 0$ is growth, $r < 0$ is decay. Used for populations, compound interest, radioactive decay, viral spread in early stages.

---

## 5. Power Law

$$
Y = a X^b
$$

Scale-free relationship — doubling the input doesn't add a fixed amount, it multiplies the output by a fixed factor. Used for city populations vs. rank (Zipf's law), earthquake magnitudes, wealth distributions, metabolic scaling.

---

## 6. Logarithmic Model

$$
Y = a + b \ln(X)
$$

Rapid change at first, then diminishing sensitivity. Used when early inputs matter most — perceived loudness vs. decibels (Weber-Fechner law), information gain, diminishing marginal utility.

---

## 7. Ordinary Differential Equation (ODE)

$$
\frac{dY}{dt} = f(Y, t)
$$

Models how a quantity changes over time as a function of its current state. Used for anything that evolves — Newton's cooling law, population dynamics, drug concentration in blood, circuit voltages.

---

## 8. System of ODEs

$$
\frac{d\mathbf{Y}}{dt} = \mathbf{f}(\mathbf{Y}, t) \quad \text{where} \quad \mathbf{Y} = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{bmatrix}
$$

Multiple interacting quantities evolving simultaneously. Used for predator-prey (Lotka-Volterra), SIR epidemics, chemical reaction networks, multi-compartment pharmacokinetics.

---

## 9. Partial Differential Equation (PDE)

$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
$$

Models quantities that change in both time and space. The example above is the heat equation. Used for heat diffusion, fluid flow, electromagnetic waves, option pricing (Black-Scholes).

---

## 10. Bayesian Model

$$
P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}
$$

Updates belief about parameters $\theta$ after observing data $D$. Prior belief times likelihood, normalized. Used when you have prior knowledge, small samples, or need to quantify uncertainty — clinical trials, A/B testing, spam filtering.

---

## 11. Markov Chain

$$
P(X_{t+1} = j \mid X_t = i) = p_{ij} \quad \text{with transition matrix} \quad \mathbf{P} = [p_{ij}]
$$

The future depends only on the present, not the past (memoryless). Used for weather modeling, PageRank, board games, customer state transitions, text generation.

---

## 12. Hidden Markov Model (HMM)

$$
\text{Hidden:} \quad P(Z_{t+1} \mid Z_t) \qquad \text{Observed:} \quad P(X_t \mid Z_t)
$$

A Markov chain you can't see directly — you infer hidden states from noisy observations. Used for speech recognition, gene sequence analysis, financial regime detection.

---

## 13. Monte Carlo Simulation

$$
\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} f(X_i) \quad \text{where} \quad X_i \sim P(X)
$$

Estimate an answer by running thousands of random experiments and averaging. Used when the math is too hard to solve analytically — option pricing, risk analysis, integration in high dimensions, physics simulations.

---

## 14. Optimization Model

$$
\min_{\theta} \; f(\theta) \quad \text{subject to} \quad g_i(\theta) \leq 0, \quad h_j(\theta) = 0
$$

Find the best parameters under constraints. Every ML training loop is this. Used for resource allocation, portfolio optimization, supply chain routing, neural network training (gradient descent).

---

## 15. Neural Network

$$
\hat{Y} = \sigma\!\Big(W_n \, \sigma\!\big(W_{n-1} \cdots \sigma(W_1 X + b_1) \cdots + b_{n-1}\big) + b_n\Big)
$$

Nested layers of linear transformations with nonlinear activations $\sigma$. Universal function approximator. Used when the relationship is too complex to specify by hand — image recognition, language models, game playing.

---

## 16. Linear Algebra / Matrix Model

$$
\mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
$$

The matrix form of the linear model. Solves all $k$ coefficients at once via $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}$. Used for multivariate regression, PCA, recommender systems, any model where data lives in a matrix.

---

## 17. Time Series (ARIMA)

$$
Y_t = c + \phi_1 Y_{t-1} + \cdots + \phi_p Y_{t-p} + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q} + \epsilon_t
$$

Past values and past errors predict the future. AR = autoregressive (past values), MA = moving average (past errors), I = integrated (differencing for stationarity). Used for stock prices, demand forecasting, sensor data, any sequential measurement.

---

## 18. Gaussian Process

$$
f(X) \sim \mathcal{GP}\!\big(m(X),\; k(X, X')\big)
$$

A distribution over functions, not parameters. Defined by a mean function $m$ and a kernel $k$ that encodes similarity. Used for Bayesian optimization, surrogate modeling, small-data regression with uncertainty quantification.

---

## 19. Agent-Based Model

$$
X_i^{(t+1)} = R\!\big(X_i^{(t)},\; \{X_j^{(t)}\}_{j \in N_i},\; \epsilon_i\big)
$$

Each agent $i$ updates its state based on its own state, its neighbors' states, and randomness. No global equation — emergent behavior arises from local rules. Used for traffic, epidemics, market dynamics, social networks, flocking.

---

## 20. Game-Theoretic Model

$$
u_i(s_i^*, s_{-i}) \geq u_i(s_i, s_{-i}) \quad \forall \; s_i \in S_i
$$

Each player $i$ picks a strategy $s_i^*$ that maximizes their utility $u_i$ given what everyone else does. Nash equilibrium: no one can improve by changing alone. Used for pricing, auctions, evolutionary biology, adversarial ML, negotiation.

---

## 21. Information-Theoretic Model

$$
H(X) = -\sum_{i} P(x_i) \log P(x_i)
$$

Measures uncertainty, information content, and compression limits. Used for feature selection (mutual information), decision trees (information gain), communication channel capacity, compression algorithms.

---

## 22. Graph / Network Model

$$
\mathbf{A} \in \{0,1\}^{n \times n} \quad \text{where} \quad A_{ij} = 1 \iff \text{edge from } i \text{ to } j
$$

Relationships encoded as nodes and edges in an adjacency matrix. Used for social networks, supply chains, knowledge graphs, molecular structures, GNNs, PageRank.

---

## 23. Survival Model

$$
S(t) = P(T > t) = e^{-\int_0^t \lambda(s)\,ds}
$$

Models time until an event — death, failure, churn. The hazard function $\lambda(t)$ is the instantaneous risk. Used for medical trials, equipment reliability, customer lifetime, insurance.

---

## 24. Stochastic Differential Equation (SDE)

$$
dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t
$$

An ODE with a noise term $dW_t$ (Brownian motion). Deterministic drift plus random shocks. Used for stock prices (Black-Scholes), molecular dynamics, noisy control systems, interest rate models.

---

## 25. Causal / Structural Equation Model (SEM)

$$
X_i = f_i(\text{Pa}(X_i),\; U_i) \quad \forall \; i
$$

Each variable is a function of its parents in the causal graph plus an unobserved noise term. Used for causal inference, mediation analysis, policy evaluation — answering "what would happen if we intervened?"



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

# Data Requirements Reference

---

## Outcome Data
*What you're predicting*

- Target variable $Y$ measured at the required granularity
- Ground truth labels (for supervised models)
- Historical outcomes for backtesting
- Human-annotated labels or expert judgments
- Event timestamps (for survival / time-to-event models)

---

## Input / Feature Data
*What feeds the model*

- Raw independent variables $X_1, \ldots, X_k$
- Derived features (ratios, differences, rolling averages)
- Categorical encodings (one-hot, ordinal, embeddings)
- Interaction terms ($X_1 \times X_2$)
- Lagged variables ($X_{t-1}, X_{t-2}, \ldots$) for time-dependent models

---

## Temporal Data
*When things happened*

- Timestamps for every observation
- Sampling frequency (seconds, minutes, daily, weekly)
- Seasonal indicators (day of week, month, quarter)
- Before/after markers for intervention or policy changes
- Duration data (time between events)

---

## Spatial / Location Data
*Where things happened*

- Coordinates (latitude, longitude)
- Region / zone / cluster labels
- Distance matrices between entities
- Adjacency or network topology
- Facility, room, rack, or asset identifiers

---

## Population / Sampling Data
*Who or what is in the dataset*

- Unique entity identifiers (user ID, asset ID, device serial)
- Group membership labels (treatment vs. control, segment, cohort)
- Inclusion / exclusion criteria documentation
- Sampling weights (if not a simple random sample)
- Census or population totals for calibration

---

## Contextual / Environmental Data
*What surrounds the observation*

- Environmental conditions (temperature, humidity, load)
- Market or economic indicators
- Policy or configuration state at time of observation
- Software version, firmware version, hardware generation
- Operator or shift information

---

## Causal / Experimental Data
*What lets you infer cause*

- Treatment assignment mechanism (random, rule-based, self-selected)
- Instrument variables for IV regression
- Confounder measurements ($Z$ variables from the causal chain)
- Pre-treatment covariates for matching
- Counterfactual or synthetic control data

---

## Quality / Metadata
*What tells you if the data is trustworthy*

- Missingness indicators (MCAR, MAR, MNAR classification)
- Data collection method and source system
- Measurement precision and resolution
- Known error rates or noise levels
- Data lineage (who created it, when, how it was transformed)
- Schema version and changelog

---

## Validation / Holdout Data
*What you test against*

- Train / validation / test split ratios
- Out-of-time holdout (future data not seen during training)
- Out-of-distribution samples (edge cases, rare events)
- Cross-validation fold assignments
- Benchmark datasets for comparison to published results

---

## Scale / Volume Requirements
*How much you need*

- Minimum sample size $n$ for statistical power
- Class balance ratios (for classification)
- Events-per-variable ratio (for regression: rule of thumb $\geq 10$)
- Storage and compute budget constraints
- Streaming vs. batch availability

---

## Legal / Ethical Data
*What you're allowed to use*

- PII inventory and anonymization status
- Data use agreements and licensing terms
- Consent records
- Regulatory constraints (GDPR, HIPAA, SOX)
- Bias audit results across protected attributes

---

## Integration / Access Data
*How you get it*

- Source system and API endpoints
- Query language (SQL, GraphQL, REST)
- Authentication and access permissions
- Refresh cadence and latency
- Format (CSV, Parquet, JSON, streaming)

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

# Parameter Estimation Methods

For Section 11 of the Modeling Journal.

---

## Analytical (Closed-Form) Methods
*Solve the math directly — one equation, one answer*

### Ordinary Least Squares (OLS)

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}
$$

Minimizes the sum of squared errors. You get an exact formula — no iteration, no guessing. Works when the model is linear in its parameters and the assumptions (normality, homoscedasticity, independence) hold.

**When to use:** Linear regression, polynomial regression, any model where the math is clean enough to invert.

---

### Maximum Likelihood Estimation (MLE)

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \; \prod_{i=1}^{n} P(y_i \mid \theta)
$$

Or equivalently in log form:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \; \sum_{i=1}^{n} \log P(y_i \mid \theta)
$$

Ask: "What parameter values make the data I actually observed most probable?" You write down the likelihood function, take the derivative, set it to zero, and solve. If you can't solve it analytically, you optimize numerically.

**When to use:** Logistic regression, Poisson models, any model where you can write down the probability of the data.

---

### Method of Moments

$$
\hat{\theta}: \quad \frac{1}{n}\sum_{i=1}^{n} g(X_i) = \mathbb{E}[g(X)]
$$

Match sample statistics (mean, variance, skewness) to their theoretical formulas and solve for the parameters. Simpler than MLE but often less efficient.

**When to use:** Quick estimates, distribution fitting, when MLE is too complex.

---

## Iterative / Numerical Methods
*The math can't be solved directly — search for the answer*

### Gradient Descent

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t)
$$

Start with a guess. Compute the slope of the loss function. Take a step downhill. Repeat. $\eta$ is the learning rate — too big and you overshoot, too small and you crawl.

**When to use:** Neural networks, any model where the loss function is differentiable but has no closed-form solution.

---

### Stochastic Gradient Descent (SGD)

$$
\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t; \, x_i, y_i)
$$

Same as gradient descent but uses one random data point (or a mini-batch) per step instead of the full dataset. Noisier but much faster on large data.

**When to use:** Large datasets, deep learning, online learning.

---

### Newton-Raphson

$$
\theta_{t+1} = \theta_t - \mathbf{H}^{-1} \nabla_\theta \mathcal{L}(\theta_t)
$$

Uses the curvature (Hessian matrix $\mathbf{H}$) to take smarter steps than gradient descent. Converges faster but each step is expensive.

**When to use:** Small-to-medium models where you can compute the Hessian — logistic regression, GLMs.

---

### Expectation-Maximization (EM)

$$
\text{E-step:} \quad Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{Z \mid X, \theta^{(t)}}[\log P(X, Z \mid \theta)]
$$

$$
\text{M-step:} \quad \theta^{(t+1)} = \arg\max_\theta \; Q(\theta \mid \theta^{(t)})
$$

For models with hidden variables. Alternate between guessing the hidden variables (E-step) and optimizing the parameters given those guesses (M-step). Guaranteed to improve each iteration.

**When to use:** Gaussian mixture models, HMMs, any model with latent/missing data.

---

## Bayesian Methods
*Combine what you already know with what the data says*

### Bayes' Theorem (Full Posterior)

$$
P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}
$$

Don't estimate a single number — estimate a whole distribution over possible parameter values. The prior $P(\theta)$ is what you believed before seeing data. The posterior $P(\theta \mid D)$ is what you believe after.

**When to use:** Small samples, strong prior knowledge, when you need uncertainty quantification, not just a point estimate.

---

### Maximum A Posteriori (MAP)

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \; P(D \mid \theta) \, P(\theta)
$$

The peak of the posterior distribution — a compromise between MLE (data only) and the prior (belief only). Equivalent to MLE with a regularization penalty.

**When to use:** When you want a single best estimate but still want to incorporate prior knowledge. Ridge regression is secretly MAP with a Gaussian prior.

---

### Markov Chain Monte Carlo (MCMC)

$$
\theta^{(t+1)} \sim q(\theta \mid \theta^{(t)}) \quad \text{accept with probability} \quad \min\!\left(1, \; \frac{P(\theta^{(t+1)} \mid D)}{P(\theta^{(t)} \mid D)}\right)
$$

When the posterior is too complex to solve analytically, generate samples from it by random walking through parameter space. After enough steps, the samples approximate the true posterior.

**When to use:** Complex Bayesian models, hierarchical models, any posterior you can evaluate but can't integrate.

---

### Variational Inference (VI)

$$
\hat{q} = \arg\min_{q \in \mathcal{Q}} \; \text{KL}\!\big(q(\theta) \;\|\; P(\theta \mid D)\big)
$$

Approximate the posterior with a simpler distribution $q$ by minimizing the divergence between them. Faster than MCMC but less exact.

**When to use:** Large-scale Bayesian models, deep generative models (VAEs), when MCMC is too slow.

---

## Regularized Methods
*Prevent overfitting by penalizing complexity*

### Ridge (L2)

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \|\boldsymbol{\beta}\|_2^2 \right\}
$$

Shrinks all parameters toward zero. Never sets them exactly to zero. Controls overfitting when you have many correlated features.

---

### Lasso (L1)

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\}
$$

Shrinks parameters toward zero AND sets some exactly to zero — automatic feature selection. Use when you suspect most variables don't matter.

---

### Elastic Net (L1 + L2)

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{ \sum_{i=1}^{n}(y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2 \right\}
$$

Best of both — selects features like Lasso, handles correlated features like Ridge.

---

## Non-Parametric Methods
*Let the data decide the shape — no fixed equation*

### Kernel Density Estimation (KDE)

$$
\hat{f}(x) = \frac{1}{n h} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right)
$$

Estimate the probability density by placing a smooth bump (kernel $K$) on each data point. Bandwidth $h$ controls smoothness.

---

### k-Nearest Neighbors (k-NN)

$$
\hat{Y}(x) = \frac{1}{k} \sum_{i \in N_k(x)} Y_i
$$

No parameters to estimate at all — just average the $k$ closest data points. The "model" is the data itself.

---

### Bootstrap

$$
\hat{\theta}^{*(b)} = T(D^{*(b)}) \quad \text{for} \quad b = 1, \ldots, B
$$

Resample your data with replacement $B$ times, estimate the parameter each time, and use the distribution of estimates to quantify uncertainty. Works for any estimator.

**When to use:** When you don't know the theoretical distribution of your estimator — confidence intervals, standard errors, bias correction.

---

## Decision Framework

```text
Can you write P(data | θ)?
├── Yes
│   ├── Can you solve dL/dθ = 0 analytically?
│   │   ├── Yes → MLE (closed-form) or OLS
│   │   └── No → Gradient Descent / Newton-Raphson
│   ├── Do you have prior knowledge?
│   │   ├── Yes → Bayesian (MAP, MCMC, VI)
│   │   └── No → MLE
│   └── Are there hidden variables?
│       └── Yes → EM Algorithm
├── No
│   ├── Can you define a loss function?
│   │   └── Yes → Gradient Descent / SGD
│   └── No → Non-parametric (KDE, k-NN, Bootstrap)
└── Too many features?
    └── Yes → Regularization (Ridge, Lasso, Elastic Net)



How will they be estimated?

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
