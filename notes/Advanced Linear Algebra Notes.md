
# MLU Mathematical Fundamentals for ML Advanced Linear Algebra Study Notes

---

## 1. Vectors and Scalars

A scalar multiple of a vector $\vec{v}$ by a positive real number $c$ produces a vector in the **same direction** as $\vec{v}$, scaled by $c$.

$$c\vec{v}, \quad c \in \mathbb{R}^+$$

Multiplying by $-1$ reverses direction. Multiplying by $0$ produces the zero vector. A scalar multiple cannot produce an orthogonal (perpendicular) vector.

---

## 2. Hadamard Product

The Hadamard product is the **element-wise** multiplication of two matrices of the same dimensions.

$$(\mathbf{A} \circ \mathbf{B})_{ij} = a_{ij} \cdot b_{ij}$$

Example:

$$\begin{bmatrix} 2 & 3 \\ 4 & 1 \end{bmatrix} \circ \begin{bmatrix} 5 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 10 & 6 \\ 4 & 3 \end{bmatrix}$$

This is NOT the standard matrix product. Think of it like two recipe sheets with the same layout — you multiply each ingredient quantity in Recipe A by the corresponding quantity in Recipe B, cell by cell.

**Why the name:** Jacques Hadamard proved the Hadamard multiplication theorem in 1899 on term-by-term products of power series. John von Neumann saw the analogy to element-wise matrix operations and suggested the name to Paul Halmos, who popularized it in his 1948 textbook *Finite-Dimensional Vector Spaces*. Also called the Schur product after Issai Schur.

---

## 3. Matrix Multiplication and Dimensions

For matrix multiplication $\mathbf{A}\mathbf{B}$, the **inner dimensions must match**.

$$\mathbf{A}_{m \times \mathbf{n}} \cdot \mathbf{B}_{\mathbf{n} \times p} = \mathbf{C}_{m \times p}$$

If $\mathbf{A}$ is $3 \times 4$ and $\mathbf{B}$ is $3 \times 2$, the inner dimensions are $4$ and $3$. They don't match. **They cannot be multiplied.**

---

## 4. Manhattan Distance

The Manhattan distance (L1 norm) between two points sums the **absolute differences** of their coordinates.

$$d_1(\vec{a}, \vec{b}) = \sum_{i=1}^{n} |b_i - a_i|$$

For $\vec{a} = (-1, -1)$ and $\vec{b} = (2, 3)$:

$$\vec{b} - \vec{a} = (3, 4)$$
$$d_1 = |3| + |4| = 7$$

Note: The Euclidean distance (L2 norm) would be $\sqrt{3^2 + 4^2} = 5$. Manhattan distance is always $\geq$ Euclidean distance.

---

## 5. Dot Product

$$\vec{u} \cdot \vec{v} = \|\vec{u}\| \|\vec{v}\| \cos\theta$$

The dot product equals the product of the magnitudes of $\vec{u}$ and $\vec{v}$ times the cosine of the angle between them. The result is a **scalar**.

### Why cosine and not sine?

The dot product measures **alignment** — how much two vectors point in the same direction. Cosine is the right function because:

| Angle | $\cos\theta$ | Meaning |
|-------|-------------|---------|
| $0°$ | $1$ | Same direction → maximum alignment |
| $90°$ | $0$ | Perpendicular → zero alignment |
| $180°$ | $-1$ | Opposite directions → maximum opposition |

If you used sine, perpendicular vectors ($90°$) would give the maximum value and same-direction vectors ($0°$) would give zero — the opposite of what alignment means.

**Shadow analogy:** The dot product is the length of the shadow one vector casts onto the other (the projection). When they're aligned, the shadow is longest. When they're perpendicular, there's no shadow at all. That shadow/projection behavior is exactly what cosine describes.

---

## 6. Cross Product vs Dot Product

| Property | Dot Product ($\vec{u} \cdot \vec{v}$) | Cross Product ($\vec{u} \times \vec{v}$) |
|----------|---------------------------------------|------------------------------------------|
| Symbol | $\cdot$ (a dot) | $\times$ (a cross) |
| Result | Scalar (single number) | Vector (new direction) |
| Measures | Alignment (how much same direction) | Perpendicularity (area spanned) |
| Trig function | $\cos\theta$ | $\sin\theta$ |
| Notation origin | Smallest symbol → simplest result | Bigger symbol → more complex result |

Both notations attributed to Josiah Willard Gibbs (1880s–1890s), who needed two visually distinct symbols for two fundamentally different kinds of vector multiplication.

### Cross Product Formula

$$\vec{u} \times \vec{v} = (u_2 v_3 - u_3 v_2, \quad u_3 v_1 - u_1 v_3, \quad u_1 v_2 - u_2 v_1)$$

Each component is built from the **other two** components' values:

- x-component → uses only y and z values from both vectors
- y-component → uses only x and z values
- z-component → uses only x and y values

Each subtraction pair like $u_2 v_3 - u_3 v_2$ measures "how much do these two components twist away from each other?" If they're proportional (parallel), the subtraction cancels to zero.

**Right-hand rule example:** Point right hand east $\vec{u} = (1,0,0)$, curl fingers north $\vec{v} = (0,1,0)$, thumb points up:

$$\vec{u} \times \vec{v} = (0 \cdot 0 - 0 \cdot 1, \quad 0 \cdot 0 - 1 \cdot 0, \quad 1 \cdot 1 - 0 \cdot 0) = (0, 0, 1)$$

Result: straight up. Perpendicular to both inputs.

---

## 7. Features (x) in Machine Learning

Features are **measurements you already have** about the thing you're trying to predict. They exist before any ML happens. You observe them, you don't compute them.

$$y = \mathbf{w} \cdot \mathbf{x} + b$$

- $\mathbf{x}$ = the actual numerical readings from the real world (sensor data, counts, measurements)
- $\mathbf{w}$ = weights the model learns from data during training
- $b$ = bias/intercept

**Datacenter example** — predicting whether a server will overheat:

- $x_1$ = current CPU temperature → read from sensor
- $x_2$ = number of jobs running → pulled from monitoring tool
- $x_3$ = ambient room temperature → read from thermostat
- $x_4$ = hours since last maintenance → checked from log

**Human analogy** — deciding if a coworker is about to quit:

- $x_1$ = how often they complain (you observe this)
- $x_2$ = how many sick days taken (you observe this)
- $x_3$ = whether they updated LinkedIn (you observe this)

The weight your brain puts on each signal ("LinkedIn update matters way more than complaining") is what the model learns. The model doesn't invent $x$. You hand it $x$ from reality, and it figures out how much each $x$ matters.

---

## 8. The Normal Equation — Full Walkthrough

$$\boldsymbol{\theta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

### Working Example: Predicting Server Temperature

3 servers, 2 features: Jobs Running and Room Temperature.

| Server | Jobs ($x_1$) | Room Temp ($x_2$) | Actual Server Temp ($y$) |
|--------|-------------|-------------------|--------------------------|
| A | 10 | 70 | 150 |
| B | 20 | 75 | 200 |
| C | 15 | 80 | 180 |

### Step 1 — Transpose $\mathbf{X}^\top$

$$\mathbf{X} = \begin{bmatrix} 10 & 70 \\ 20 & 75 \\ 15 & 80 \end{bmatrix} \quad \Rightarrow \quad \mathbf{X}^\top = \begin{bmatrix} 10 & 20 & 15 \\ 70 & 75 & 80 \end{bmatrix}$$

The transpose is a **mirror flip along the diagonal** — rows become columns and columns become rows. Not a rotation. The value at position (row $i$, column $j$) moves to (row $j$, column $i$).

Now each row is a feature looking across all servers:

- Row 1 = Jobs: $[10, 20, 15]$ across all servers
- Row 2 = Room Temp: $[70, 75, 80]$ across all servers

### Step 2 — $\mathbf{X}^\top\mathbf{X}$ (Feature Relationship Map)

Every cell is a dot product between two features:

$$\mathbf{X}^\top\mathbf{X} = \begin{bmatrix} 10 \cdot 10 + 20 \cdot 20 + 15 \cdot 15 & 10 \cdot 70 + 20 \cdot 75 + 15 \cdot 80 \\ 70 \cdot 10 + 75 \cdot 20 + 80 \cdot 15 & 70 \cdot 70 + 75 \cdot 75 + 80 \cdot 80 \end{bmatrix}$$

$$= \begin{bmatrix} 725 & 3400 \\ 3400 & 16925 \end{bmatrix}$$

What each cell means:

- **Diagonal** (725, 16925) = each feature dotted with itself → how spread out is this feature? (variance)
- **Off-diagonal** (3400) = features dotted with each other → do they move together? (covariance/overlap)

### Step 3 — $\mathbf{X}^\top\mathbf{y}$ (Raw Testimonies)

$$\mathbf{X}^\top\mathbf{y} = \begin{bmatrix} 10 \cdot 150 + 20 \cdot 200 + 15 \cdot 180 \\ 70 \cdot 150 + 75 \cdot 200 + 80 \cdot 180 \end{bmatrix} = \begin{bmatrix} 8200 \\ 39900 \end{bmatrix}$$

- Jobs testimony: 8,200 → "When I was high, server temp was high"
- Room Temp testimony: 39,900 → "When I was high, server temp was high too"

Room Temp's number is 5x bigger, but that's **misleading**. Room Temp values are 70–80 while Jobs are 10–20. Bigger numbers in, bigger numbers out. Like comparing salaries in Yen vs Dollars.

### Step 4 — $(\mathbf{X}^\top\mathbf{X})^{-1}$ (The Corrective Lens)

$$(\mathbf{X}^\top\mathbf{X})^{-1} = \begin{bmatrix} 0.023817 & -0.004785 \\ -0.004785 & 0.001020 \end{bmatrix}$$

The **negative off-diagonal values** are the key. They subtract the overlap between features that move together.

### Step 5 — $\boldsymbol{\theta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ (The Fair Weights)

**Jobs weight ($\theta_1$):**

$$\theta_1 = (0.023817 \times 8200) + (-0.004785 \times 39900)$$
$$= 195.30 - 190.90 = 4.40$$

The 195.30 is Jobs' raw credit. The $-190.90$ subtracts Room Temp's overlapping influence. Almost all the raw signal gets subtracted because these features moved together.

**Room Temp weight ($\theta_2$):**

$$\theta_2 = (-0.004785 \times 8200) + (0.001020 \times 39900)$$
$$= -39.23 + 40.71 = 1.47$$

The $-39.23$ subtracts Jobs' overlap. The $40.71$ is Room Temp's raw credit.

**Translation:** Every additional job adds ~$4.4°F$. Every additional degree of room temp adds ~$1.5°F$.

---

## 9. With vs Without the Corrective Lens

### Without (Naive — ignore overlap between features)

$$\theta_1^{naive} = \frac{X^\top y[\text{jobs}]}{X^\top X[\text{jobs,jobs}]} = \frac{8200}{725} = 11.31$$

$$\theta_2^{naive} = \frac{X^\top y[\text{temp}]}{X^\top X[\text{temp,temp}]} = \frac{39900}{16925} = 2.36$$

| Server | Naive Prediction | Actual | Off by |
|--------|-----------------|--------|--------|
| A | 278 | 150 | +128 |
| B | 403 | 200 | +203 |
| C | 358 | 180 | +178 |

Both features take full credit for the same temperature rise. Predictions massively overshoot.

### With (The Inverse — untangle overlap)

| Server | Corrected Prediction | Actual | Off by |
|--------|---------------------|--------|--------|
| A | 147 | 150 | -3 |
| B | 199 | 200 | -1 |
| C | 184 | 180 | +4 |

| | Naive | With Lens |
|---|---|---|
| Jobs weight | 11.31 (inflated) | 4.40 (fair) |
| RoomTemp weight | 2.36 (inflated) | 1.47 (fair) |

**Group project analogy:** Two teammates — Alex (10–20 hours) and Blake (70–80 hours) — both claim credit for the grade. Blake's raw testimony is 5x bigger because Blake logged more hours. But Blake was in the library at the same time as Alex every single time. The naive approach gives both full credit and the "predicted grades" blow up to impossible numbers. The inverse is the professor pulling them into separate meetings and subtracting the hours they were just sitting next to each other. The negative off-diagonal values in the inverse are literally the subtraction of co-presence.

**Critical limitation (multicollinearity):** If two features always move together, the model cannot reliably separate their individual effects. The weights become unstable. No amount of math can tell you who was really coding if Alex and Blake always show up together.

**The fix — feature engineering:** Add features that break the lockstep. Instead of just "hours in the library," track git commits, lines of code, files touched, PR reviews. Now Alex has 2–5 commits while Blake has 45–60. The lockstep is broken and the model can finally see who did the real work. Choosing which features to measure is where domain expertise becomes the real superpower.

---

## 10. OLS Lab — Predicting Checkout Amount

### Business Problem

Predict customer Checkout Amount from Spend in Category and Prime Member duration on Amazon's gardening product page.

### Design Matrix (the column of 1s trick)

$$\mathbf{X} = \begin{bmatrix} 1 & x_1^{(1)} \\ 1 & x_1^{(2)} \\ \vdots & \vdots \\ 1 & x_1^{(n)} \end{bmatrix}$$

The column of 1s handles the bias term $w_0$ automatically inside the matrix math. The prediction equation becomes:

$$\hat{y} = \mathbf{X} \cdot \mathbf{w}$$

### Numerical Example (4 customers, univariate)

| Customer | Spend ($x_1$) | Checkout ($y$) |
|----------|--------------|----------------|
| 1 | \$200 | \$180 |
| 2 | \$50 | \$60 |
| 3 | \$400 | \$350 |
| 4 | \$150 | \$140 |

**Design matrix with 1s column:**

$$\mathbf{X} = \begin{bmatrix} 1 & 200 \\ 1 & 50 \\ 1 & 400 \\ 1 & 150 \end{bmatrix}$$

**$\mathbf{X}^\top\mathbf{X}$:**

$$\begin{bmatrix} 4 & 800 \\ 800 & 225000 \end{bmatrix}$$

- $[0,0] = 4$ → count of customers
- $[0,1] = 800$ → total sum of all spending
- $[1,1] = 225000$ → sum of squares of spending

**$\mathbf{X}^\top\mathbf{y}$:**

$$\begin{bmatrix} 730 \\ 200000 \end{bmatrix}$$

- $730$ → sum of all checkout amounts
- $200000$ → each customer's spend $\times$ their checkout, summed

**Solution:** $w_0 = 16.35$ (bias), $w_1 = 0.831$ (slope)

$$\text{Checkout} = 0.831 \times \text{Spend} + \$16.35$$

For every \$1 more in past spending, checkout goes up ~\$0.83, plus a baseline of \$16.35.

| Customer | Spend | Predicted | Actual |
|----------|-------|-----------|--------|
| 1 | \$200 | \$182.50 | \$180 |
| 2 | \$50 | \$57.90 | \$60 |
| 3 | \$400 | \$348.70 | \$350 |
| 4 | \$150 | \$141.00 | \$140 |

### Multivariate Extension

Adding Prime Member as a second feature:

$$\text{Checkout} = w_1 \times \text{Spend} + w_2 \times \text{Prime Member} + w_0$$

The design matrix now has 3 columns. Geometrically: fitting a plane through 3D data instead of a line through 2D data. The key question: does adding Prime Member improve predictions on unseen test data?

---

## 11. Evaluation Metrics

### $R^2$ (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- $R^2 = 1.0$ → perfect fit
- $R^2 = 0.0$ → model is no better than predicting the mean
- Context-dependent: physics expects $>0.9$, consumer behavior $0.3$–$0.5$ is decent

### MSE (Mean Squared Error)

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Lower = better. Units are squared (e.g., dollars$^2$).

Always evaluate on **test data** (unseen). Good training + poor test = overfitting = a tech who can only fix racks they've seen before.

---

## 12. What OLS Gives You and What It Doesn't

**OLS gives you:** predictions for linear relationships, weight values showing direction and magnitude of each feature's influence, $R^2$ and MSE to measure model quality, a baseline to compare every other model against.

**OLS doesn't give you:** statistical significance (need p-values, t-tests — use `statsmodels.OLS`), non-linear patterns (need polynomial regression, trees, neural nets), reliable feature importance when features are correlated (need VIF, SHAP), overfitting protection (need Ridge L2, Lasso L1), scalability to massive data (need gradient descent — the normal equation requires matrix inversion which is expensive at scale).

**The progression:** OLS is the transistor. Regularized regression adds overfitting protection. Statistical testing tells you if a feature actually matters or got lucky. Gradient descent replaces the normal equation when data gets huge. Neural networks are stacked linear models with non-linear activations between them.

---

## 13. Key Formulas — Quick Reference

| Concept | Formula |
|---------|---------|
| Dot product | $\vec{u} \cdot \vec{v} = \|\vec{u}\|\|\vec{v}\|\cos\theta$ |
| Cross product (x-component) | $u_2 v_3 - u_3 v_2$ |
| Manhattan distance | $\sum \|b_i - a_i\|$ |
| Hadamard product | $(\mathbf{A} \circ \mathbf{B})_{ij} = a_{ij} \cdot b_{ij}$ |
| Normal equation | $\boldsymbol{\theta} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| Linear prediction | $\hat{y} = \mathbf{X}\mathbf{w}$ |
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| 2×2 matrix inverse | $\frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$ |

---

*Module 1 of MLU Mathematical Fundamentals for ML. One commit a day.*
