# Mathematical Fundamentals for Machine Learning Study Notes

Program vs AI

A traditional program follows rules a human wrote explicitly. If experience is greater than 5 and degree equals Masters then salary equals 90000. The logic is fixed. A human must rewrite the code to change behavior. It may or may not use matrices. A spreadsheet is a matrix. Nobody calls Excel AI.

An AI system learns its own rules from data. You give it 10,000 examples and it discovers the pattern itself. The learned knowledge is stored in matrices of weights that no human wrote. Feed it new data and it updates itself. The defining characteristic is the learning loop. See data, make a prediction, check how wrong you were, adjust the weights, repeat millions of times.

The difference is not the presence of matrices. Both can use them. The difference is whether the matrices were written by a human or learned by the machine.

---

## Course Overview

This covers three mathematical pillars — linear algebra, differential calculus, and probability/statistics — and connects each to concrete ML and generative AI applications. The final project is building a complete collaborative filtering recommender system. Approximately 20 hours. Prerequisites: Python, Jupyter notebooks, AWS console familiarity.

---

## PILLAR 1: LINEAR ALGEBRA — "The Skeleton"

Linear algebra is how machines see and organize data. Every image, sentence, or dataset gets turned into matrices and vectors before a model ever touches it.

Linear Regression

Linear regression is a mathematical model that predicts a continuous number by adding together weighted inputs along a best fit line that minimizes prediction error.

Etymology

"Linear" from Latin linearis meaning "of a line." The model predicts by adding scaled variables together in a straight line equation. Addition is the defining operation.

"Regression" from Latin regressus meaning "to go back." Coined by Sir Francis Galton in the 1880s when he observed that tall parents produce slightly shorter children and short parents produce slightly taller children. The data regressed, went back, toward the average. The name stuck for all line fitting ever since.

Characteristics of Linear Regression

The model is a weighted sum. ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ Each input contributes independently through addition, no multiplication between variables, no exponents. The output is always a continuous number, a price, a temperature, a salary, never a category. The weights reveal how much each input matters and by how much. It serves two purposes. Prediction of new outcomes and understanding which variables move the needle.

### Core Data Structures

Vectors are single lists of numbers (like arrays in computer science). Matrices have 2 indices (rows and columns). Tensors go beyond 2 dimensions (3+ indices). These structures store and represent large datasets efficiently.

$$\text{Vector: } \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n \qquad \text{Matrix: } \mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

$$\text{Tensor: } \mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k} \quad (k \text{ indices})$$

### The Four Transformations

All four can happen simultaneously in a single matrix multiplication.

STRETCH: Multiply a column by a number. Doubling the Sales column pulls every data point away from center vertically. In ML, normalizing or scaling features is stretching/compressing so a model doesn't think salary=80,000 matters more than years_experience=5 just because the number is bigger.

$$\text{Stretch by } \alpha: \quad \begin{bmatrix} \alpha & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \alpha x \\ y \end{bmatrix}$$

COMPRESS: Multiply by a fraction. The opposite of stretch. PCA finds directions where data barely varies and compresses them to near-zero, effectively dropping useless dimensions. Like realizing column 37 in a 50-column spreadsheet is basically the same number for every row.

$$\text{Compress by } \frac{1}{\alpha}: \quad \begin{bmatrix} \frac{1}{\alpha} & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \frac{x}{\alpha} \\ y \end{bmatrix}, \quad 0 < \frac{1}{\alpha} < 1$$

ROTATE: Tilt the entire coordinate system. PCA rotates your 50 columns into 50 new columns ranked by importance, then you keep only the top 5 or 10. The relationships between data points stay identical — like turning a map sideways.

$$\text{Rotate by } \theta: \quad \mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{bmatrix}$$

COMBINE: Create a new column by mixing existing columns. New_Column = (0.8 × Hours) + (0.6 × Sales). Neural network layers do exactly this — take input columns, multiply by weights, add together to make new columns. That is matrix multiplication.

$$\text{Combine: } \quad z = w_1 x_1 + w_2 x_2 = \begin{bmatrix} w_1 & w_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \mathbf{w}^\top \mathbf{x}$$

$$\text{General form — one matrix encodes all four: } \quad \mathbf{y} = \mathbf{W}\mathbf{x}$$

### Linear Operations in ML

MULTIPLICATION: The fundamental building block. A single matrix multiply encodes stretch, rotate, compress, and combine all at once.

$$\mathbf{C} = \mathbf{A}\mathbf{B}, \quad C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

ADDITION: Shift data, combine signals. Adding a bias term in a neural network: output = weights × input + bias.

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

SUBTRACTION: Find differences, compute errors. Loss calculation: error = predicted − actual.

$$\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$$

TRANSPOSITION: Flip rows and columns. Computing $\mathbf{X}^\top \mathbf{X}$ in linear regression.

$$\mathbf{A}^\top_{ij} = \mathbf{A}_{ji} \qquad \text{i.e., rows become columns}$$

DOT PRODUCT: Measure similarity between two vectors. Cosine similarity — how search engines and LLMs decide "king" is more similar to "queen" than to "banana."

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

INVERSION: "Undo" a transformation. Solving linear regression directly: $\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$.

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I} \qquad \text{where } \mathbf{I} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$\text{OLS closed-form: } \quad \hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

ELEMENT-WISE OPERATIONS: Multiply/add matching entries one-by-one. Attention masks in transformers — zeroing out positions so the model cannot see future tokens.

$$(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} \cdot B_{ij} \qquad \text{(Hadamard product)}$$

$$\text{Masked attention: } \quad \text{score}_{ij} = \text{score}_{ij} \odot M_{ij}, \quad M_{ij} \in \{0, 1\}$$

### Basis

A basis is your coordinate system — the rulers you use to describe where things are. Every vector can be represented as a linear combination of the basis elements. This means you can reach any point by mixing the basis directions together.

$$\mathbf{v} = c_1 \mathbf{e}_1 + c_2 \mathbf{e}_2 + \cdots + c_n \mathbf{e}_n = \sum_{i=1}^{n} c_i \mathbf{e}_i$$

$$\text{Standard basis in } \mathbb{R}^3: \quad \mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

GPS ANALOGY: Your GPS describes every location using North/South and East/West. Those two directions are a basis. Any location can be described as "go X miles North + Y miles East." You could pick different directions (Northeast/Southeast) and it would still work — that is a different basis for the same space.

NON-COLLINEAR REQUIREMENT: Collinear means pointing in the same or exactly opposite direction. If both your basis vectors point North, you can only reach places on a single line — you can never go East. For a basis to work, no vector can be redundant (something you could already make by combining the others).

$$\text{Collinear (bad): } \mathbf{b}_2 = k\,\mathbf{b}_1 \implies \text{span}\{\mathbf{b}_1, \mathbf{b}_2\} = \text{a line, not a plane}$$

$$\text{Linear independence (good): } \sum_{i=1}^{n} c_i \mathbf{b}_i = \mathbf{0} \implies c_1 = c_2 = \cdots = c_n = 0$$

WHY THIS MATTERS IN ML: Feature redundancy — if column 3 equals column 1 + column 2, it is collinear and adds zero information. This causes multicollinearity in regression and makes the math blow up. PCA finds a new basis where each direction captures maximum independent information. Embeddings use each dimension as an independent aspect of meaning. The rank of a matrix (number of truly independent columns) tells you how much real information your data contains.

$$\text{rank}(\mathbf{A}) = \text{number of linearly independent columns of } \mathbf{A}$$

### Coefficients

A coefficient is the number you multiply by. In "Alice = 10 × Hours_direction + 50 × Sales_direction," the 10 and 50 are coefficients. They tell you how much of each basis direction to use.

$$\mathbf{v} = \underbrace{10}_{\text{coeff}_1} \cdot \mathbf{e}_{\text{hours}} + \underbrace{50}_{\text{coeff}_2} \cdot \mathbf{e}_{\text{sales}}$$

RECIPE ANALOGY: "2 cups flour + 1 cup sugar + 0.5 cups butter." The numbers 2, 1, and 0.5 are coefficients. The ingredients are your basis. Any baked good is a linear combination of ingredients.

ETYMOLOGY: "Co-" means together/with (Latin: cum). "Efficient" from Latin efficiens meaning accomplishing. Coefficient literally means "the thing that works together with" the variable. Term entered mathematics in the 1600s from François Viète.

IN ML: When a model learns weights, it is learning coefficients. A large coefficient means "this feature matters a lot." A near-zero coefficient means "this feature is almost irrelevant."

$$\hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \sum_{i=1}^{n} w_i x_i + b$$

### Vector Addition

Combining two lists element by element: $[10, 50] + [20, 100] = [30, 150]$.

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

IN ML: Residual connections in transformers: $\text{output} = f(\mathbf{x}) + \mathbf{x}$. Gradient updates: $\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \nabla \mathcal{L}$. Word embeddings: $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$.

### Matrix Addition/Subtraction Dimension Requirement

Addition is element-by-element. Each cell in Matrix A gets added to the corresponding cell in Matrix B. If there is no corresponding cell, the operation is meaningless.

$$\mathbf{A} + \mathbf{B} \text{ is defined } \iff \mathbf{A} \in \mathbb{R}^{m \times n} \text{ and } \mathbf{B} \in \mathbb{R}^{m \times n}$$

$$\mathbb{R}^{2 \times 3} + \mathbb{R}^{2 \times 4} = \text{undefined}$$

STORE ANALOGY: Store A tracks Shoes/Hats/Bags (2×3 matrix). Store B tracks Shoes/Hats/Bags/Scarves (2×4 matrix). What do you add to the Scarves column? There is nothing there. It is not zero — it is undefined. Like trying to merge two forms where one has 3 fields and the other has 4.

### Handling Mismatched Dimensions in Practice

PADDING: Add rows/columns of zeros until dimensions match. LLMs pad shorter sentences with a PAD token and use attention masks to ignore the padding.

$$\mathbf{A} \in \mathbb{R}^{2 \times 3} \xrightarrow{\text{pad}} \mathbf{A}' = \begin{bmatrix} a_{11} & a_{12} & a_{13} & 0 \\ a_{21} & a_{22} & a_{23} & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}$$

TRUNCATION: Cut off extra rows/columns.

BROADCASTING (NumPy/PyTorch): Automatically stretch a smaller matrix to match a bigger one by repeating values.

$$\text{Broadcasting: } \quad \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + 5 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 5 \\ 5 \\ 5 \end{bmatrix} = \begin{bmatrix} 6 \\ 7 \\ 8 \end{bmatrix}$$

EMBEDDING/PROJECTION: Transform data into a common dimension.

IMPUTATION: Fill missing values with mean, median, or predicted value.

### Scalars

A scalar is a single number, as opposed to a vector (list) or matrix (grid). The word comes from "scale" because multiplying by a single number scales things up or down.

$$\alpha \in \mathbb{R} \quad \text{(a scalar — zero indices, just a number)}$$

SCALAR MULTIPLICATION: Multiply every element by the same number. $3 \times [10, 50, 25] = [30, 150, 75]$. In ML, the learning rate is a scalar controlling step size.

$$\alpha \mathbf{v} = \alpha \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} \alpha v_1 \\ \alpha v_2 \\ \alpha v_3 \end{bmatrix}$$

SCALAR ADDITION: Add the same number to every element. The bias in a neural network is scalar/vector addition shifting output up or down.

### Vector Space

A vector space is any collection of things where addition and scaling behave sensibly. A room where you can move freely — go from any point to any other (addition), speed up or slow down (scalar multiplication), and never fall out of the room (closure).

$$\text{A vector space } V \text{ over } \mathbb{R} \text{ satisfies: } \forall\, \mathbf{u}, \mathbf{v} \in V,\; \forall\, \alpha, \beta \in \mathbb{R}:$$

THE 8 AXIOMS:

$$\begin{aligned}
1.\quad & \mathbf{u} + \mathbf{v} \in V & \text{(closure under addition)} \\
2.\quad & \alpha \mathbf{u} \in V & \text{(closure under scalar multiplication)} \\
3.\quad & (\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w}) & \text{(associativity)} \\
4.\quad & \mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u} & \text{(commutativity)} \\
5.\quad & \exists\, \mathbf{0} \in V : \mathbf{v} + \mathbf{0} = \mathbf{v} & \text{(zero vector)} \\
6.\quad & \forall\, \mathbf{v}, \exists\, (-\mathbf{v}) : \mathbf{v} + (-\mathbf{v}) = \mathbf{0} & \text{(additive inverse)} \\
7.\quad & \alpha(\beta \mathbf{v}) = (\alpha \beta)\mathbf{v} & \text{(compatibility)} \\
8.\quad & 1 \cdot \mathbf{v} = \mathbf{v} & \text{(multiplicative identity)}
\end{aligned}$$

TEST: Can I add these things together and scale them, and do I always stay in the same kind of thing?

EXAMPLES: RGB colors — yes. Word embeddings — yes. Positive numbers only — no (scale 5 by −1 and you get −5, outside the set). Images as pixel matrices — yes.

WHY IT MATTERS: If your data lives in a vector space, all of linear algebra's tools work on it. This is why ML's first step is almost always converting data into vectors.

---

## PILLAR 2: DIFFERENTIAL CALCULUS — "The Steering Wheel"

Calculus tells the model which direction to adjust to get better. When a model makes a wrong prediction, calculus computes the gradient — a slope that says "move this way to reduce your error."

BLINDFOLDED LANDSCAPE ANALOGY: You are blindfolded on a hilly landscape trying to find the lowest valley. You can feel the slope under your feet. Calculus IS that sense of slope. Gradient descent is the strategy: always step downhill.

### Key Applications

GRADIENT DESCENT: The core training loop of nearly every neural network. Compute the derivative of the loss function, nudge weights in the opposite direction.

$$\text{Derivative: } \quad f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

$$\text{Gradient (multi-variable): } \quad \nabla \mathcal{L}(\mathbf{w}) = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w_1} \\ \frac{\partial \mathcal{L}}{\partial w_2} \\ \vdots \\ \frac{\partial \mathcal{L}}{\partial w_n} \end{bmatrix}$$

$$\text{Weight update rule: } \quad \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$$

$$\text{where } \eta = \text{learning rate (scalar), } \nabla \mathcal{L} = \text{gradient of the loss}$$

LINEAR AND LOGISTIC REGRESSION: The "Hello World" of ML. Calculus finds the best-fit line (linear) or best-fit S-curve (logistic) by minimizing error.

$$\text{Linear regression loss (MSE): } \quad \mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)^2 = \frac{1}{N} \|\hat{\mathbf{y}} - \mathbf{y}\|^2$$

$$\text{Logistic (sigmoid): } \quad \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

$$\text{Binary cross-entropy loss: } \quad \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$

BACKPROPAGATION IN LLMs: Gradients flowing backward through billions of parameters.

$$\text{Chain rule: } \quad \frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial a_3} \cdot \frac{\partial a_3}{\partial a_2} \cdot \frac{\partial a_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1}$$

$$\text{Each layer passes its local gradient backward — the chain rule composes them.}$$

---

## PILLAR 3: PROBABILITY AND STATISTICS — "The Intuition Engine"

Probability is how models deal with uncertainty. Statistics gives tools to extract signal from noise and quantify confidence.

DETECTIVE ANALOGY: You never have perfect evidence — just clues. Probability tells you how to weigh clues. Statistics tells you when you have enough clues to make a call.

### Key Applications

MAXIMUM LIKELIHOOD ESTIMATION (MLE): Given data, what parameters make that data most probable? Like asking: if I assume this coin is biased, what bias best explains the flips I have seen?

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} \; P(\text{data} \mid \theta) = \arg\max_{\theta} \prod_{i=1}^{N} P(x_i \mid \theta)$$

$$\text{In practice, maximize the log-likelihood: } \quad \hat{\theta} = \arg\max_{\theta} \sum_{i=1}^{N} \log P(x_i \mid \theta)$$

A/B TESTING: Did this new feature actually improve click-through, or was it random noise?

$$\text{Null hypothesis } H_0: \mu_A = \mu_B \qquad \text{Alternative } H_1: \mu_A \neq \mu_B$$

$$\text{Test statistic: } \quad t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}} \qquad \text{Reject } H_0 \text{ if } p < \alpha$$

FRAUD DETECTION: Model what normal transactions look like, then flag outliers.

LLM NEXT-TOKEN GENERATION: When GPT picks the next word, it samples from a probability distribution over the entire vocabulary. Temperature, top-k, top-p are all probability concepts.

$$P(w_t \mid w_1, \ldots, w_{t-1}) = \text{softmax}\!\left(\frac{\mathbf{z}}{\tau}\right)_t = \frac{e^{z_t / \tau}}{\sum_{j=1}^{|V|} e^{z_j / \tau}}$$

$$\text{where } \tau = \text{temperature}, \; |V| = \text{vocabulary size}, \; \mathbf{z} = \text{logits}$$

$$\tau \to 0: \text{ deterministic (always pick highest)} \qquad \tau \to \infty: \text{ uniform random}$$

---

## WORD EMBEDDINGS — Deep Dive

### How Words Get Their Vectors

Nobody hand-picks the numbers. The model learns them by reading billions of sentences. Start with random numbers. Feed billions of sentences. The model predicts missing words, gets it wrong, computes loss, and gradient descent nudges the vectors. Over time, words appearing in similar contexts drift toward similar vectors.

$$\mathbf{v}_{\text{word}} \in \mathbb{R}^{d} \quad \text{where } d = \text{embedding dimension (e.g., 768)}$$

$$\text{Objective (Skip-gram): } \quad \max_{\theta} \sum_{t=1}^{T} \sum_{-c \leq j \leq c,\, j \neq 0} \log P(w_{t+j} \mid w_t;\, \theta)$$

THE COMPANY YOU KEEP ANALOGY: Figuring out someone's personality by only observing who they hang out with. You never meet them directly, but if they are always seen with doctors, at hospitals, wearing scrubs — you infer they are probably in medicine. The company a word keeps IS its vector.

### What Each Dimension Means

There is no clean human label for most dimensions. The model found its own basis. Researchers discovered that DIRECTIONS in the space encode meaning ($\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$), but these directions are smeared across many dimensions.

SMOOTHIE ANALOGY: You blended strawberry, banana, and spinach. You can taste it is fruity and earthy, but you cannot point to one sip and say "that is the strawberry molecule." Meaning is distributed across the whole vector.

### The "No Close Friends" Insight

A word/person that appears in many different contexts but never deeply in any sits near the center of the space, has low similarity with everything, and is hard to predict. The ML term is "low information" or "high entropy."

$$H(X) = -\sum_{x} P(x) \log P(x) \qquad \text{(Shannon entropy — high when distribution is flat)}$$

The word "the" is like the person with no close friends — it appears next to every word. Its embedding is uninformative. Contrast with "scalpel" which almost exclusively hangs out with surgery, doctor, operating room. Its vector is sharp and specific.

COGNITIVE PARALLEL: Someone with no deep connections would be hard to embed — hard to place in a social map. Maps to identity diffusion in psychology — a state where someone has not committed to a clear self-concept. Their personality vector has high variance and low magnitude.

[FLAGGED FOR FUTURE EXPLORATION]

### Cosine Similarity — Measuring Distance

$$\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \; \|\mathbf{B}\|} = \frac{\displaystyle\sum_{i=1}^{n} A_i B_i}{\sqrt{\displaystyle\sum_{i=1}^{n} A_i^2} \;\cdot\; \sqrt{\displaystyle\sum_{i=1}^{n} B_i^2}}$$

A and B are the complete vectors of the two words being compared. The dot product multiplies matching positions between two DIFFERENT vectors and sums them. If numbers tend to have the same sign, the dot product is large and positive (vectors point the same way). If signs clash, the dot product is small or negative.

WORKED EXAMPLE:

$$\vec{\text{king}} = \begin{bmatrix} 2 \\ 3 \\ 1 \end{bmatrix}, \quad \vec{\text{banana}} = \begin{bmatrix} 1 \\ -2 \\ 4 \end{bmatrix}$$

$$\text{Dot product} = (2)(1) + (3)(-2) + (1)(4) = 2 - 6 + 4 = 0$$

$$\|\vec{\text{king}}\| = \sqrt{4 + 9 + 1} = \sqrt{14} \approx 3.74, \quad \|\vec{\text{banana}}\| = \sqrt{1 + 4 + 16} = \sqrt{21} \approx 4.58$$

$$\text{cosine\_similarity} = \frac{0}{3.74 \times 4.58} = 0.0 \quad \text{(perpendicular — completely unrelated)}$$

WORKED EXAMPLE:

$$\vec{\text{king}} = \begin{bmatrix} 2 \\ 3 \\ 1 \end{bmatrix}, \quad \vec{\text{queen}} = \begin{bmatrix} 2 \\ 2 \\ 1 \end{bmatrix}$$

$$\text{Dot product} = 4 + 6 + 1 = 11, \quad \|\vec{\text{queen}}\| = \sqrt{4 + 4 + 1} = 3.0$$

$$\text{cosine\_similarity} = \frac{11}{3.74 \times 3.0} \approx 0.98 \quad \text{(almost identical)}$$

PERSONALITY SURVEY ANALOGY: Two people fill out a 768-question survey. The dot product checks: did they answer similarly on each question?

Typical values: king-queen ≈ 0.75, king-man ≈ 0.55, king-banana ≈ 0.05, king-king = 1.00.

Similar words are close but never identical. "Big" and "large" might be 0.90+ but still not identical because they are used in slightly different ways.

### Why 768 Dimensions?

768 is one of the most common embedding dimensions. BERT-base and GPT-2 small both use 768. It comes from 12 attention heads × 64 dimensions per head = 768. An engineering choice, not a mathematical law. More dimensions = more capacity but more computation. 768 is a sweet spot.

$$d_{\text{model}} = n_{\text{heads}} \times d_{\text{head}} = 12 \times 64 = 768$$

Other models: Word2Vec = 300, GPT-3 = 12,288, GPT-4 estimated 12,000+.

### Why You Cannot Create a Universal Formula for Vectors

The input problem: What properties of a word do you feed in? Letters fail ("king" and "kong" share 3/4 letters but mean different things; "buy" and "purchase" share zero letters but mean the same thing). Prefixes/suffixes help but most meaning is not in morphology ("bank" the river and "bank" the financial institution are structurally identical). Dictionary definitions create circular dependency — you need vectors to process the definition.

FUNDAMENTAL PROBLEM: Meaning comes from how a word is used in relation to all other words across billions of sentences. The training process IS the formula — it is just very expensive to compute.

NAME-TO-PERSONALITY ANALOGY: Can you write a formula that takes a person's name and outputs their personality? No — personality comes from a lifetime of experiences. The name alone does not contain enough information. Same with words.

WHAT DOES EXIST: FastText uses subword information to generate vectors for unseen words. Distillation trains smaller models to mimic larger ones. But both still require training.

### Why You Cannot Hardcode Universal Vectors

Distances are not universal — they depend on training data. A model trained on medieval history places "king" near "sword" and "castle." A model trained on modern news places "king" near "government" and "policy."

Dimensions are entangled and interdependent (the smoothie problem). Two models might encode the same relationships using completely different internal representations.

All 50,000+ word relationships must be consistent simultaneously. It is a massive jigsaw puzzle — every piece constrains every other piece.

WEDDING SEATING ANALOGY: Seating 50,000 people where everyone has preferences about who they sit near. You cannot place two people and call it done — every placement affects every other.

### Subword Tokenization

Modern models break text into tokens — chunks that are often prefixes, suffixes, or syllables. "unhappiness" becomes ["un", "happiness"]. "playing" becomes ["play", "ing"]. The prefix "un-" develops a vector encoding negation across all words. The suffix "-ing" encodes ongoing action.

$$\text{"unhappiness"} \xrightarrow{\text{tokenize}} [\text{"un"}, \text{"happi"}, \text{"ness"}] \xrightarrow{\text{embed}} [\mathbf{v}_{\text{un}}, \mathbf{v}_{\text{happi}}, \mathbf{v}_{\text{ness}}]$$

### Contextual Embeddings (The Banana King Problem)

Modern models do not use one fixed vector per word. They use contextual embeddings — the vector for "king" changes depending on the sentence. "The king wore a crown" produces a royalty-region vector. "The banana king of Dole" shifts toward business + fruit. "King cobra" shifts toward animals + danger.

$$\mathbf{v}_{\text{king}}^{(\text{context})} = f_{\text{transformer}}(\text{"The king wore a crown"})[\text{pos}_{\text{king}}]$$

$$\mathbf{v}_{\text{king}}^{(\text{royalty})} \neq \mathbf{v}_{\text{king}}^{(\text{business})} \neq \mathbf{v}_{\text{king}}^{(\text{animal})}$$

The attention mechanism does this — when the model sees "banana king," attention pulls the contextual representation of "king" toward the banana/business region for that sentence only.

### AI Creativity

AI can combine known concepts in novel ways (compositional creativity). It cannot invent concepts with zero basis in training data. AI creativity is like DJing — remixing existing tracks in novel combinations — rather than composing from silence.

---

## TRANSFORMERS, GPT, AND TIME

### What Is GPT?

GPT = Generative Pre-trained Transformer. Generative means it creates text one word at a time. Pre-trained means it was trained on massive data before you talk to it. Transformer is the specific architecture.

### The Core Problem: Word Order

Transformers read all words at once (unlike RNNs which read sequentially). This means "the dog bit the man" and "the man bit the dog" would look identical without a solution.

### Positional Encoding

Add a unique position tag vector to each word's embedding. "dog" embedding + position_2_vector = "dog at position 2." Uses sine and cosine waves so nearby positions have similar encodings and distant positions have different ones.

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\mathbf{x}_{\text{input}} = \mathbf{v}_{\text{word}} + \mathbf{PE}_{\text{pos}}$$

TIMESTAMP ANALOGY: 10 shuffled security camera photos are meaningless. Add a timestamp to each and you can reconstruct the sequence without experiencing the time yourself.

For predicting the future: the model looks at positions 1 through N (the past), notices patterns, and generates position N+1. Like reading a story to chapter 9 and guessing chapter 10.

### Sine and Cosine

Draw a circle with radius 1. Pick any point on the circle. Sine = how high the point is (vertical position). Cosine = how far right the point is (horizontal position). As you walk around the circle, both wave smoothly between −1 and +1.

$$\sin^2\theta + \cos^2\theta = 1 \qquad \text{(always on the unit circle)}$$

$$\sin(0) = 0, \quad \cos(0) = 1 \qquad \sin\!\left(\frac{\pi}{2}\right) = 1, \quad \cos\!\left(\frac{\pi}{2}\right) = 0$$

CLOCK ANALOGY: Watch the tip of the second hand. At 12 o'clock: sine=1, cosine=0. At 3 o'clock: sine=0, cosine=1. At 6 o'clock: sine=−1, cosine=0. At 9 o'clock: sine=0, cosine=−1.

Why transformers use them: Every position gets a unique combination. Nearby positions have similar values. The pattern repeats at different scales (like having both a second hand and hour hand).

DATACENTER ANALOGY: Each rack has a unique hum — a chord of multiple frequencies. You could navigate blindfolded by listening. That is positional encoding.

### RNNs vs Transformers

RNNs read one word at a time sequentially, carrying a hidden state forward. Time is built into the architecture. Problem: by step 500, the model has mostly forgotten step 1 (vanishing gradient problem — like a game of telephone).

$$\text{RNN: } \quad \mathbf{h}_t = \tanh(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})$$

$$\text{Vanishing gradient: } \quad \frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \to 0 \text{ as } T \to \infty$$

Transformers see all words at once in parallel. Attention lets each word look at every other word directly. "Crown" can attend to "king" even 200 words apart. No telephone game. This is why transformers won.

$$\text{Self-Attention: } \quad \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

$$\text{where } \mathbf{Q} = \text{queries}, \; \mathbf{K} = \text{keys}, \; \mathbf{V} = \text{values}, \; d_k = \text{key dimension}$$

### The Time Axiom Problem

Pure vector spaces cannot model time naturally. The zero vector axiom (do nothing) and inverse axiom (reverse any action) are violated by time — you cannot freeze or reverse time.

$$\text{Zero vector axiom: } \exists\, \mathbf{0} : \mathbf{v} + \mathbf{0} = \mathbf{v} \quad \xrightarrow{\text{time}} \quad \text{"freeze time"? — not physical}$$

$$\text{Inverse axiom: } \forall\, \mathbf{v}, \exists\, (-\mathbf{v}) : \mathbf{v} + (-\mathbf{v}) = \mathbf{0} \quad \xrightarrow{\text{time}} \quad \text{"reverse time"? — not physical}$$

ML SOLUTIONS: Snapshots — freeze time into slices, each row is a vector in a valid vector space. Positional encoding — stamp each data point with its place in sequence. RNNs — build the arrow of time into architecture, not math. Neural ODEs — model rates of change using calculus for truly continuous time.

$$\text{Neural ODE: } \quad \frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t) \qquad \text{(continuous-time hidden state)}$$

KEY INSIGHT: Vector spaces model the STATE of a system at a moment, not the flow of time itself. Time is handled by architecture, loss functions, and sequential structure.

MIND PALACE ANALOGY: Each room in a datacenter is a snapshot — a vector space. The hallway connecting Room 1 to Room 2 to Room 3 is the arrow of time. The hallway is one-way. Rooms obey vector space rules; the hallway does not need to.

---

## CROSS-LINGUAL MODEL COMPARISON — Research Direction

[FLAGGED FOR FUTURE EXPLORATION]

Training models on different languages produces different internal geometries. The same concept gets carved up differently across languages.

EXAMPLES: Russian has separate words for light blue and dark blue — Russian-trained models treat these as distinct concepts. Russian speakers are measurably faster at distinguishing them in perception tests. Mandarin uses vertical metaphors for time; English uses horizontal. Japanese has multiple words for "I" encoding gender, formality, and social status.

RESEARCH POTENTIAL: Compare embedding geometries across languages to reveal which concepts are universal, which are culturally constructed, and how social structures are encoded. Tools exist (multilingual BERT, XLM-RoBERTa). Methodology is straightforward (Procrustes alignment). This is publishable-quality research at an active frontier.

$$\text{Procrustes alignment: } \quad \min_{\mathbf{R}} \|\mathbf{X}\mathbf{R} - \mathbf{Y}\|_F^2 \quad \text{s.t. } \mathbf{R}^\top\mathbf{R} = \mathbf{I}$$

$$\text{where } \mathbf{X} = \text{embeddings in language A}, \; \mathbf{Y} = \text{embeddings in language B}$$

CONNECTS TO: Sapir-Whorf hypothesis — does language shape thought or does thought shape language? AI provides empirical testing ground.

---

## HOW AI REASONS

AI does not search a database of pre-written answers. It learned patterns at multiple levels during training: word patterns (which words appear near which), structural patterns (how explanations are built), reasoning patterns (logical argument chains), analogy patterns (domain mapping), and meta-patterns (adapting style to the learner).

Novel questions are answered by composing learned patterns in real-time. Like a jazz musician who has internalized scales, chord progressions, and emotional arcs, then improvises by combining them.

HONEST LIMITATION: AI does not "understand" the way humans do. It has sophisticated pattern matching and composition. Whether that constitutes reasoning or understanding is an open question. The gap between pattern composition and conscious reasoning is exactly what Cognitive ML explores.

The quality of answers is directly proportional to the quality of questions. Questions that activate deep patterns (like "would the time axiom break vector spaces") force composition across multiple knowledge domains and produce sharper answers than surface-level retrieval questions.

---

## JAPANESE LANGUAGE LEARNING VIA AI — Prompt Templates

PATTERN DISCOVERY: "What are the 5 most common sentence structures in Japanese? Show me the pattern as a formula, then 3 examples of each."

GRAMMAR AS FUNCTIONS: "Treat Japanese particles as functions. wa = topic_marker(noun), wo = object_marker(noun), ni = direction_marker(noun). Give me 20 sentences where I can see these functions applied."

$$\text{wa}(\text{noun}) \to \text{topic}, \quad \text{wo}(\text{noun}) \to \text{object}, \quad \text{ni}(\text{noun}) \to \text{direction}$$

$$\text{Sentence} = \text{wa}(\text{watashi}) + \text{wo}(\text{sushi}) + \text{verb}(\text{taberu}) \to \text{"I eat sushi"}$$

REVERSE ENGINEERING: "Give me 15 Japanese sentences with literal word-by-word translations. Let me reverse-engineer the grammar rules myself."

POLITENESS LEVELS: "Show me the same idea at 4 politeness levels. Highlight ONLY the parts that change."

GENERATIVE RULES: "What are the 10 generative rules of Japanese grammar that let me construct any basic sentence? Think of it like 10 composable functions."

---

## FINAL PROJECT: RECOMMENDER SYSTEM

Everything converges. Linear algebra represents users and items as vectors/matrices and factors the rating matrix. Calculus uses gradient descent to optimize predictions. Probability models uncertainty in ratings and evaluates with statistical metrics.

$$\text{Rating matrix: } \quad \mathbf{R} \approx \mathbf{U} \mathbf{V}^\top, \quad \mathbf{U} \in \mathbb{R}^{m \times k}, \; \mathbf{V} \in \mathbb{R}^{n \times k}$$

$$\text{Predicted rating: } \quad \hat{r}_{ij} = \mathbf{u}_i^\top \mathbf{v}_j = \sum_{\ell=1}^{k} u_{i\ell} \, v_{j\ell}$$

$$\text{Loss: } \quad \mathcal{L} = \sum_{(i,j) \in \text{observed}} (r_{ij} - \mathbf{u}_i^\top \mathbf{v}_j)^2 + \lambda(\|\mathbf{U}\|_F^2 + \|\mathbf{V}\|_F^2)$$

$$\text{Update: } \quad \mathbf{u}_i \leftarrow \mathbf{u}_i - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{u}_i}, \qquad \mathbf{v}_j \leftarrow \mathbf{v}_j - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{v}_j}$$

ENGINE ANALOGY: Algebra is the metal frame, calculus is the fuel injection system, probability is the sensor array telling you how well it is running.

---

## KEY ANALOGIES INDEX (for quick review)

- Linear algebra = the skeleton of ML
- Calculus = the steering wheel / blindfolded on a hill
- Probability = the detective weighing clues
- Stretch/compress/rotate/combine = photo editor filters
- Basis = GPS coordinate system (North/East)
- Coefficients = recipe amounts
- Non-collinear = North AND East (not North AND North)
- Word embeddings = personality inferred from who you hang out with
- Dimensions of a vector = smoothie (flavors distributed, not separable)
- Cosine similarity = flashlight beams pointing same/different directions
- Matrix dimension mismatch = merging forms with different numbers of fields
- Training vectors = seating 50,000 people at a wedding
- Vector space = a room you can move freely in without falling out
- Positional encoding = timestamps on shuffled photos
- Sine/cosine = clock second hand position
- Time in ML = rooms (vector spaces) connected by one-way hallways
- AI creativity = DJing (remixing) not composing from silence
- Scalar = the manager giving everyone the same multiplier
- No close friends = the word "the" (appears everywhere, means little)


## SYMBOL INDEX

### Greek Letters

| Symbol | Name | Meaning in These Notes |
|--------|------|----------------------|
| $\alpha$ | alpha | Scalar multiplier; stretch/compress factor; significance level in hypothesis testing |
| $\beta$ | beta | Scalar multiplier (used with $\alpha$ in vector space axioms) |
| $\eta$ | eta | Learning rate — scalar controlling gradient descent step size |
| $\theta$ | theta | Rotation angle; model parameters; angle between two vectors |
| $\lambda$ | lambda | Regularization strength in the recommender loss function |
| $\mu$ | mu | Population mean (used in A/B testing hypotheses: $\mu_A$, $\mu_B$) |
| $\sigma$ | sigma | Sigmoid activation function: $\sigma(z) = \frac{1}{1+e^{-z}}$ |
| $\tau$ | tau | Temperature parameter in softmax sampling |
| $\ell$ | ell | Summation index over latent factors in matrix factorization |

### Greek — Uppercase

| Symbol | Name | Meaning in These Notes |
|--------|------|----------------------|
| $\Delta x$ | Delta x | Small change in $x$ (used in the derivative limit definition) |
| $\Sigma$ (via $\sum$) | Sigma | Summation operator |
| $\Pi$ (via $\prod$) | Pi | Product operator (used in MLE and vanishing gradient) |

### Calligraphic & Special Typefaces

| Symbol | Name | Meaning in These Notes |
|--------|------|----------------------|
| $\mathcal{L}$ | Script L | Loss function — the error the model is trying to minimize |
| $\mathcal{T}$ | Script T | Tensor (multi-dimensional array) |
| $\mathbb{R}$ | Blackboard R | The set of all real numbers |
| $\mathbb{R}^n$ | R-n | $n$-dimensional real vector space |
| $\mathbb{R}^{m \times n}$ | R-m-by-n | Space of all real $m$-row, $n$-column matrices |
| $\mathbf{I}$ | Bold I | Identity matrix (the "do nothing" matrix: $\mathbf{AI} = \mathbf{A}$) |

### Vectors & Matrices (Bold / Arrow Notation)

| Symbol | Meaning |
|--------|---------|
| $\mathbf{v}$, $\mathbf{u}$, $\mathbf{w}$, $\mathbf{x}$, $\mathbf{y}$, $\mathbf{z}$, $\mathbf{b}$, $\mathbf{h}$, $\mathbf{e}$ | Vectors (lowercase bold) |
| $\mathbf{A}$, $\mathbf{B}$, $\mathbf{C}$, $\mathbf{W}$, $\mathbf{X}$, $\mathbf{R}$, $\mathbf{U}$, $\mathbf{V}$, $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{M}$ | Matrices (uppercase bold) |
| $\vec{\text{king}}$, $\vec{\text{queen}}$, $\vec{\text{banana}}$ | Word embedding vectors (arrow notation) |
| $\mathbf{v}_{\text{word}}$ | Embedding vector for a specific word |
| $\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3$ | Standard basis vectors |
| $\mathbf{0}$ | Zero vector |
| $\mathbf{u}_i$, $\mathbf{v}_j$ | User $i$ and item $j$ latent factor vectors (recommender system) |

### Operators & Operations

| Symbol | Name | Meaning |
|--------|------|---------|
| $+$, $-$ | Addition, Subtraction | Element-wise vector/matrix addition and subtraction |
| $\cdot$ | Dot (scalar product) | $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$ — measures similarity |
| $\odot$ | Hadamard product | Element-wise multiplication: $(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} B_{ij}$ |
| $\times$ | Times / Cross | Scalar multiplication or dimensional notation ($m \times n$) |
| $\mathbf{A}^\top$ | Transpose | Flip rows and columns: $A^\top_{ij} = A_{ji}$ |
| $\mathbf{A}^{-1}$ | Inverse | The matrix that "undoes" $\mathbf{A}$: $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ |
| $\|\mathbf{v}\|$ | Norm (magnitude) | Length of a vector: $\sqrt{\sum v_i^2}$ |
| $\|\cdot\|_F$ | Frobenius norm | Matrix "length": $\sqrt{\sum_{ij} A_{ij}^2}$ — used in regularization |
| $\nabla$ | Nabla / Del | Gradient operator — vector of all partial derivatives |
| $\frac{\partial}{\partial w}$ | Partial derivative | Rate of change of a function with respect to one variable |
| $\frac{d}{dt}$ | Total derivative | Rate of change with respect to time (Neural ODE) |
| $\lim$ | Limit | Approaching a value (used in derivative definition) |
| $\sum$ | Summation | Add up a sequence of terms |
| $\prod$ | Product | Multiply a sequence of terms |
| $\arg\max$ | Argmax | The input value that maximizes a function |
| $\min$ | Minimum | The smallest value (used in Procrustes alignment) |
| $\log$ | Logarithm | Natural log — converts products to sums (log-likelihood) |
| $\exp$ / $e^{(\cdot)}$ | Exponential | Euler's number raised to a power (softmax, sigmoid) |
| $\sqrt{\cdot}$ | Square root | Used in norms, cosine similarity denominator, attention scaling |

### Subscripts & Superscripts

| Notation | Meaning |
|----------|---------|
| $A_{ij}$ | Element at row $i$, column $j$ of matrix $\mathbf{A}$ |
| $v_i$ | The $i$-th element of vector $\mathbf{v}$ |
| $w_t$ | Word at position $t$ in a sequence |
| $\mathbf{w}_{t+1}$ | Weights after update step $t+1$ |
| $\mathbf{h}_t$ | Hidden state at time step $t$ (RNN) |
| $\hat{y}$ | Predicted value (the "hat" means estimate) |
| $\hat{\theta}_{\text{MLE}}$ | Maximum likelihood estimate of parameter $\theta$ |
| $\hat{\mathbf{w}}$ | Estimated / learned weight vector |
| $r_{ij}$ | Observed rating by user $i$ for item $j$ |
| $\hat{r}_{ij}$ | Predicted rating by user $i$ for item $j$ |
| $d_{\text{model}}$ | Total embedding dimension (e.g., 768) |
| $d_k$ | Key dimension in attention ($= d_{\text{model}} / n_{\text{heads}}$) |
| $n_{\text{heads}}$ | Number of attention heads |

### Set & Logic Notation

| Symbol | Name | Meaning |
|--------|------|---------|
| $\in$ | Element of | "$\mathbf{v} \in \mathbb{R}^n$" means $\mathbf{v}$ is a vector in $n$-dimensional real space |
| $\forall$ | For all | Universal quantifier — "for every" |
| $\exists$ | There exists | Existential quantifier — "there is at least one" |
| $\implies$ | Implies | Logical consequence — "if ... then ..." |
| $\iff$ | If and only if | Equivalence — both directions hold |
| $\{0, 1\}$ | Set | A collection of elements (here: binary mask values) |
| $\to$ | Approaches / Maps to | Limit direction or function mapping |

### Functions & Named Operations

| Symbol | Name | Definition / Role |
|--------|------|-------------------|
| $\sin$, $\cos$ | Sine, Cosine | Trigonometric functions — used in positional encoding and rotation |
| $\tanh$ | Hyperbolic tangent | Activation function in RNNs: squashes values to $(-1, 1)$ |
| $\text{softmax}$ | Softmax | Converts logits to probabilities: $\frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| $\text{span}\{\cdot\}$ | Span | All vectors reachable by combining the given set |
| $\text{rank}(\cdot)$ | Rank | Number of linearly independent columns in a matrix |
| $f_\theta$ | Parameterized function | A neural network with learnable parameters $\theta$ |
| $f_{\text{transformer}}$ | Transformer function | Maps a sentence to contextual embedding vectors |
| $PE_{(pos, i)}$ | Positional encoding | Sine/cosine tag for position $pos$, dimension $i$ |
| $H(X)$ | Shannon entropy | Measures uncertainty: $-\sum P(x)\log P(x)$ |

### Key Composite Expressions (Quick Reference)

| Expression | What It Computes | Section |
|------------|-----------------|---------|
| $\mathbf{z} = \mathbf{Wx} + \mathbf{b}$ | Neural network linear layer | Linear Operations |
| $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ | OLS regression closed-form solution | Inversion |
| $\frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|\|\mathbf{B}\|}$ | Cosine similarity | Cosine Similarity |
| $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\nabla\mathcal{L}$ | Gradient descent update | Gradient Descent |
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | Sigmoid / logistic function | Logistic Regression |
| $-\frac{1}{N}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Binary cross-entropy loss | Logistic Regression |
| $\frac{e^{z_t/\tau}}{\sum_j e^{z_j/\tau}}$ | Temperature-scaled softmax | LLM Next-Token |
| $\hat{\theta} = \arg\max_\theta \sum \log P(x_i \mid \theta)$ | Maximum likelihood estimation | MLE |
| $\text{softmax}\!\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}}\right)\mathbf{V}$ | Scaled dot-product attention | Self-Attention |
| $\mathbf{R} \approx \mathbf{UV}^\top$ | Matrix factorization (recommender) | Final Project |
| $\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), t)$ | Neural ODE | Time Axiom |
| $\min_\mathbf{R}\|\mathbf{XR}-\mathbf{Y}\|_F^2$ s.t. $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$ | Procrustes alignment | Cross-Lingual |
| $H(X) = -\sum P(x)\log P(x)$ | Shannon entropy | No Close Friends |

