
# Mathematical Fundamentals for Machine Learning — Study Notes
### Caesar Anyogu | September 27, 2026
### Source: Conversational deep-dive with AI assistant

---

## Course Overview

This course covers three mathematical pillars — linear algebra, differential calculus, and probability/statistics — and connects each to concrete ML and generative AI applications. The final project is building a complete collaborative filtering recommender system. Approximately 20 hours. Prerequisites: Python, Jupyter notebooks, AWS console familiarity.

---

## PILLAR 1: LINEAR ALGEBRA — "The Skeleton"

Linear algebra is how machines see and organize data. Every image, sentence, or dataset gets turned into matrices and vectors before a model ever touches it.

### Core Data Structures

Vectors are single lists of numbers (like arrays in computer science). Matrices have 2 indices (rows and columns). Tensors go beyond 2 dimensions (3+ indices). These structures store and represent large datasets efficiently.

### The Four Transformations

All four can happen simultaneously in a single matrix multiplication.

STRETCH: Multiply a column by a number. Doubling the Sales column pulls every data point away from center vertically. In ML, normalizing or scaling features is stretching/compressing so a model doesn't think salary=80,000 matters more than years_experience=5 just because the number is bigger.

COMPRESS: Multiply by a fraction. The opposite of stretch. PCA finds directions where data barely varies and compresses them to near-zero, effectively dropping useless dimensions. Like realizing column 37 in a 50-column spreadsheet is basically the same number for every row.

ROTATE: Tilt the entire coordinate system. PCA rotates your 50 columns into 50 new columns ranked by importance, then you keep only the top 5 or 10. The relationships between data points stay identical — like turning a map sideways.

COMBINE: Create a new column by mixing existing columns. New_Column = (0.8 x Hours) + (0.6 x Sales). Neural network layers do exactly this — take input columns, multiply by weights, add together to make new columns. That is matrix multiplication.

### Linear Operations in ML

MULTIPLICATION: The fundamental building block. A single matrix multiply encodes stretch, rotate, compress, and combine all at once.

ADDITION: Shift data, combine signals. Adding a bias term in a neural network: output = weights x input + bias.

SUBTRACTION: Find differences, compute errors. Loss calculation: error = predicted - actual.

TRANSPOSITION: Flip rows and columns. Computing X-transpose times X in linear regression.

DOT PRODUCT: Measure similarity between two vectors. Cosine similarity — how search engines and LLMs decide "king" is more similar to "queen" than to "banana."

INVERSION: "Undo" a transformation. Solving linear regression directly: weights = (X-transpose X)^-1 X-transpose y.

ELEMENT-WISE OPERATIONS: Multiply/add matching entries one-by-one. Attention masks in transformers — zeroing out positions so the model cannot see future tokens.

### Basis

A basis is your coordinate system — the rulers you use to describe where things are. Every vector can be represented as a linear combination of the basis elements. This means you can reach any point by mixing the basis directions together.

GPS ANALOGY: Your GPS describes every location using North/South and East/West. Those two directions are a basis. Any location can be described as "go X miles North + Y miles East." You could pick different directions (Northeast/Southeast) and it would still work — that is a different basis for the same space.

NON-COLLINEAR REQUIREMENT: Collinear means pointing in the same or exactly opposite direction. If both your basis vectors point North, you can only reach places on a single line — you can never go East. For a basis to work, no vector can be redundant (something you could already make by combining the others).

WHY THIS MATTERS IN ML: Feature redundancy — if column 3 equals column 1 + column 2, it is collinear and adds zero information. This causes multicollinearity in regression and makes the math blow up. PCA finds a new basis where each direction captures maximum independent information. Embeddings use each dimension as an independent aspect of meaning. The rank of a matrix (number of truly independent columns) tells you how much real information your data contains.

### Coefficients

A coefficient is the number you multiply by. In "Alice = 10 x Hours_direction + 50 x Sales_direction," the 10 and 50 are coefficients. They tell you how much of each basis direction to use.

RECIPE ANALOGY: "2 cups flour + 1 cup sugar + 0.5 cups butter." The numbers 2, 1, and 0.5 are coefficients. The ingredients are your basis. Any baked good is a linear combination of ingredients.

ETYMOLOGY: "Co-" means together/with (Latin: cum). "Efficient" from Latin efficiens meaning accomplishing. Coefficient literally means "the thing that works together with" the variable. Term entered mathematics in the 1600s from François Viete.

IN ML: When a model learns weights, it is learning coefficients. A large coefficient means "this feature matters a lot." A near-zero coefficient means "this feature is almost irrelevant."

### Vector Addition

Combining two lists element by element: [10, 50] + [20, 100] = [30, 150].

IN ML: Residual connections in transformers: output = layer(input) + input. Gradient updates: new_weights = old_weights + (learning_rate x gradient). Word embeddings: king - man + woman ≈ queen.

### Matrix Addition/Subtraction Dimension Requirement

Addition is element-by-element. Each cell in Matrix A gets added to the corresponding cell in Matrix B. If there is no corresponding cell, the operation is meaningless.

STORE ANALOGY: Store A tracks Shoes/Hats/Bags (2x3 matrix). Store B tracks Shoes/Hats/Bags/Scarves (2x4 matrix). What do you add to the Scarves column? There is nothing there. It is not zero — it is undefined. Like trying to merge two forms where one has 3 fields and the other has 4.

### Handling Mismatched Dimensions in Practice

PADDING: Add rows/columns of zeros until dimensions match. LLMs pad shorter sentences with a PAD token and use attention masks to ignore the padding.

TRUNCATION: Cut off extra rows/columns.

BROADCASTING (NumPy/PyTorch): Automatically stretch a smaller matrix to match a bigger one by repeating values.

EMBEDDING/PROJECTION: Transform data into a common dimension.

IMPUTATION: Fill missing values with mean, median, or predicted value.

### Scalars

A scalar is a single number, as opposed to a vector (list) or matrix (grid). The word comes from "scale" because multiplying by a single number scales things up or down.

SCALAR MULTIPLICATION: Multiply every element by the same number. 3 x [10, 50, 25] = [30, 150, 75]. In ML, the learning rate is a scalar controlling step size.

SCALAR ADDITION: Add the same number to every element. The bias in a neural network is scalar/vector addition shifting output up or down.

### Vector Space

A vector space is any collection of things where addition and scaling behave sensibly. A room where you can move freely — go from any point to any other (addition), speed up or slow down (scalar multiplication), and never fall out of the room (closure).

THE 8 AXIOMS: Closure under addition (adding two vectors gives another vector in the same space). Closure under scalar multiplication. Associativity of addition. Commutativity of addition. Identity element/zero vector (a "do nothing" vector exists). Inverse elements (every vector has an opposite). Compatibility of scalar multiplication. Identity of scalar multiplication (1 x V = V).

TEST: Can I add these things together and scale them, and do I always stay in the same kind of thing?

EXAMPLES: RGB colors — yes. Word embeddings — yes. Positive numbers only — no (scale 5 by -1 and you get -5, outside the set). Images as pixel matrices — yes.

WHY IT MATTERS: If your data lives in a vector space, all of linear algebra's tools work on it. This is why ML's first step is almost always converting data into vectors.

---

## PILLAR 2: DIFFERENTIAL CALCULUS — "The Steering Wheel"

Calculus tells the model which direction to adjust to get better. When a model makes a wrong prediction, calculus computes the gradient — a slope that says "move this way to reduce your error."

BLINDFOLDED LANDSCAPE ANALOGY: You are blindfolded on a hilly landscape trying to find the lowest valley. You can feel the slope under your feet. Calculus IS that sense of slope. Gradient descent is the strategy: always step downhill.

### Key Applications

GRADIENT DESCENT: The core training loop of nearly every neural network. Compute the derivative of the loss function, nudge weights in the opposite direction.

LINEAR AND LOGISTIC REGRESSION: The "Hello World" of ML. Calculus finds the best-fit line (linear) or best-fit S-curve (logistic) by minimizing error.

BACKPROPAGATION IN LLMs: Gradients flowing backward through billions of parameters.

---

## PILLAR 3: PROBABILITY AND STATISTICS — "The Intuition Engine"

Probability is how models deal with uncertainty. Statistics gives tools to extract signal from noise and quantify confidence.

DETECTIVE ANALOGY: You never have perfect evidence — just clues. Probability tells you how to weigh clues. Statistics tells you when you have enough clues to make a call.

### Key Applications

MAXIMUM LIKELIHOOD ESTIMATION (MLE): Given data, what parameters make that data most probable? Like asking: if I assume this coin is biased, what bias best explains the flips I have seen?

A/B TESTING: Did this new feature actually improve click-through, or was it random noise?

FRAUD DETECTION: Model what normal transactions look like, then flag outliers.

LLM NEXT-TOKEN GENERATION: When GPT picks the next word, it samples from a probability distribution over the entire vocabulary. Temperature, top-k, top-p are all probability concepts.

---

## WORD EMBEDDINGS — Deep Dive

### How Words Get Their Vectors

Nobody hand-picks the numbers. The model learns them by reading billions of sentences. Start with random numbers. Feed billions of sentences. The model predicts missing words, gets it wrong, computes loss, and gradient descent nudges the vectors. Over time, words appearing in similar contexts drift toward similar vectors.

THE COMPANY YOU KEEP ANALOGY: Figuring out someone's personality by only observing who they hang out with. You never meet them directly, but if they are always seen with doctors, at hospitals, wearing scrubs — you infer they are probably in medicine. The company a word keeps IS its vector.

### What Each Dimension Means

There is no clean human label for most dimensions. The model found its own basis. Researchers discovered that DIRECTIONS in the space encode meaning (king - man + woman ≈ queen), but these directions are smeared across many dimensions.

SMOOTHIE ANALOGY: You blended strawberry, banana, and spinach. You can taste it is fruity and earthy, but you cannot point to one sip and say "that is the strawberry molecule." Meaning is distributed across the whole vector.

### The "No Close Friends" Insight

A word/person that appears in many different contexts but never deeply in any sits near the center of the space, has low similarity with everything, and is hard to predict. The ML term is "low information" or "high entropy."

The word "the" is like the person with no close friends — it appears next to every word. Its embedding is uninformative. Contrast with "scalpel" which almost exclusively hangs out with surgery, doctor, operating room. Its vector is sharp and specific.

COGNITIVE PARALLEL: Someone with no deep connections would be hard to embed — hard to place in a social map. Maps to identity diffusion in psychology — a state where someone has not committed to a clear self-concept. Their personality vector has high variance and low magnitude.

[FLAGGED FOR FUTURE EXPLORATION]

### Cosine Similarity — Measuring Distance

cosine_similarity = (A dot B) / (|A| x |B|)

A and B are the complete vectors of the two words being compared. The dot product multiplies matching positions between two DIFFERENT vectors and sums them. If numbers tend to have the same sign, the dot product is large and positive (vectors point the same way). If signs clash, the dot product is small or negative.

WORKED EXAMPLE: king = [2, 3, 1], banana = [1, -2, 4]. Dot product = (2x1) + (3x-2) + (1x4) = 2 + -6 + 4 = 0. Lengths: |king| = sqrt(14) ≈ 3.74, |banana| = sqrt(21) ≈ 4.58. Result: 0 / (3.74 x 4.58) = 0.0. King and banana are perpendicular — completely unrelated.

WORKED EXAMPLE: king = [2, 3, 1], queen = [2, 2, 1]. Dot product = 4 + 6 + 1 = 11. Result: 11 / (3.74 x 3.0) = 0.98. Almost identical.

PERSONALITY SURVEY ANALOGY: Two people fill out a 768-question survey. The dot product checks: did they answer similarly on each question?

Typical values: king-queen ≈ 0.75, king-man ≈ 0.55, king-banana ≈ 0.05, king-king = 1.00.

Similar words are close but never identical. "Big" and "large" might be 0.90+ but still not identical because they are used in slightly different ways.

### Why 768 Dimensions?

768 is one of the most common embedding dimensions. BERT-base and GPT-2 small both use 768. It comes from 12 attention heads x 64 dimensions per head = 768. An engineering choice, not a mathematical law. More dimensions = more capacity but more computation. 768 is a sweet spot.

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

### Contextual Embeddings (The Banana King Problem)

Modern models do not use one fixed vector per word. They use contextual embeddings — the vector for "king" changes depending on the sentence. "The king wore a crown" produces a royalty-region vector. "The banana king of Dole" shifts toward business + fruit. "King cobra" shifts toward animals + danger.

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

TIMESTAMP ANALOGY: 10 shuffled security camera photos are meaningless. Add a timestamp to each and you can reconstruct the sequence without experiencing the time yourself.

For predicting the future: the model looks at positions 1 through N (the past), notices patterns, and generates position N+1. Like reading a story to chapter 9 and guessing chapter 10.

### Sine and Cosine

Draw a circle with radius 1. Pick any point on the circle. Sine = how high the point is (vertical position). Cosine = how far right the point is (horizontal position). As you walk around the circle, both wave smoothly between -1 and +1.

CLOCK ANALOGY: Watch the tip of the second hand. At 12 o'clock: sine=1, cosine=0. At 3 o'clock: sine=0, cosine=1. At 6 o'clock: sine=-1, cosine=0. At 9 o'clock: sine=0, cosine=-1.

Why transformers use them: Every position gets a unique combination. Nearby positions have similar values. The pattern repeats at different scales (like having both a second hand and hour hand).

DATACENTER ANALOGY: Each rack has a unique hum — a chord of multiple frequencies. You could navigate blindfolded by listening. That is positional encoding.

### RNNs vs Transformers

RNNs read one word at a time sequentially, carrying a hidden state forward. Time is built into the architecture. Problem: by step 500, the model has mostly forgotten step 1 (vanishing gradient problem — like a game of telephone).

Transformers see all words at once in parallel. Attention lets each word look at every other word directly. "Crown" can attend to "king" even 200 words apart. No telephone game. This is why transformers won.

### The Time Axiom Problem

Pure vector spaces cannot model time naturally. The zero vector axiom (do nothing) and inverse axiom (reverse any action) are violated by time — you cannot freeze or reverse time.

ML SOLUTIONS: Snapshots — freeze time into slices, each row is a vector in a valid vector space. Positional encoding — stamp each data point with its place in sequence. RNNs — build the arrow of time into architecture, not math. Neural ODEs — model rates of change using calculus for truly continuous time.

KEY INSIGHT: Vector spaces model the STATE of a system at a moment, not the flow of time itself. Time is handled by architecture, loss functions, and sequential structure.

MIND PALACE ANALOGY: Each room in a datacenter is a snapshot — a vector space. The hallway connecting Room 1 to Room 2 to Room 3 is the arrow of time. The hallway is one-way. Rooms obey vector space rules; the hallway does not need to.

---

## CROSS-LINGUAL MODEL COMPARISON — Research Direction

[FLAGGED FOR FUTURE EXPLORATION]

Training models on different languages produces different internal geometries. The same concept gets carved up differently across languages.

EXAMPLES: Russian has separate words for light blue and dark blue — Russian-trained models treat these as distinct concepts. Russian speakers are measurably faster at distinguishing them in perception tests. Mandarin uses vertical metaphors for time; English uses horizontal. Japanese has multiple words for "I" encoding gender, formality, and social status.

RESEARCH POTENTIAL: Compare embedding geometries across languages to reveal which concepts are universal, which are culturally constructed, and how social structures are encoded. Tools exist (multilingual BERT, XLM-RoBERTa). Methodology is straightforward (Procrustes alignment). This is publishable-quality research at an active frontier.

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

REVERSE ENGINEERING: "Give me 15 Japanese sentences with literal word-by-word translations. Let me reverse-engineer the grammar rules myself."

POLITENESS LEVELS: "Show me the same idea at 4 politeness levels. Highlight ONLY the parts that change."

GENERATIVE RULES: "What are the 10 generative rules of Japanese grammar that let me construct any basic sentence? Think of it like 10 composable functions."

---

## FINAL PROJECT: RECOMMENDER SYSTEM

Everything converges. Linear algebra represents users and items as vectors/matrices and factors the rating matrix. Calculus uses gradient descent to optimize predictions. Probability models uncertainty in ratings and evaluates with statistical metrics.

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

