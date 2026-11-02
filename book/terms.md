
# AI/ML Terminology

This glossary defines the foundational concepts introduced across the machine learning and artificial intelligence lessons.

```{glossary}
1D-Convolutional Neural Network (1D-CNN)
    A deep learning architecture highly specialized for processing one-dimensional sequence data, such as raw time-series radio frequency (RF) signals in Electronic Warfare.

A* Search
    An optimal and complete informed search algorithm that selects the path minimizing the total estimated cost, calculated by adding the exact cost incurred so far and the estimated cost to the target ($f(n) = g(n) + h(n)$).

Action
    A specific, localized possibility for an agent to move to another state or manipulate its environment (e.g., move North, fire thruster, block IP).

Action (ReAct)
    The step in the ReAct loop where the agent executes a specific, structured tool call (e.g., querying a database or calling an API) to interact with the external environment.

Action-Value Function
    The mathematical expected return achieved after taking a specific action $a$ in state $s$, and then strictly following policy $\pi$ for all subsequent steps. It is used to directly compare different actions available in the same state.

Activation Function
    The mathematical "gate" in an artificial neuron that determines whether, and how strongly, the neuron fires. It introduces non-linearity into neural networks. *(Examples: Sigmoid, ReLU).*

Adam (Optimizer)
    A widely used variant of gradient descent that combines momentum (a running average of recent gradients) with a separate, adaptive step size for each weight. It is the common default optimizer for training deep networks.

Adjudicator Agent
    In multi-agent wargaming, an impartial referee AI that receives proposed actions from competing agents (Red/Blue teams), evaluates them against simulated physical rules or probabilities, and updates the shared environment.

Admissible Heuristic
    A heuristic that never overestimates the true cost to reach the goal ($0 \le h(n) \le h^*(n)$). If a heuristic is admissible, A* applied to a Search Tree is mathematically guaranteed to find the optimal path.

Adversarial Evasion
    A deployment-phase attack where an adversary subtly alters the physical environment or input data (e.g., using digital static or specialized camouflage) to mathematically force an already-trained AI model to make a highly confident, incorrect prediction.

Adversarial Example
    An input that has been deliberately, and often imperceptibly, modified so that a trained model makes a confident but incorrect prediction (e.g., faint added noise that causes an image classifier to label a panda as a gibbon). The basic tool behind {term}`Adversarial Evasion`.

Adversarial Search
    A class of search algorithms designed for competitive environments where multiple agents have conflicting goals, typically modeled as a zero-sum game.

Agent
    A system that processes information and produces an output from an input. It perceives its environment and takes actions to achieve a specific goal.

Agentic AI
    An AI system endowed with agency: the capacity to autonomously plan, reason, use tools, and take actions over multiple steps to achieve a complex goal, rather than simply reacting to a single prompt.

Algorithmic Bias
    A failure mode where an AI system produces systematically prejudiced or skewed results, usually caused by imbalanced, incomplete, or historically biased training data.

Ancestor (Bayesian Network)
    In a Bayesian network graph, any node along an upstream directed path that leads to a given target node. If directed edges connect $A \to B \to C$, then $A$ is an ancestor of both $B$ and $C$.

Anomaly Detection
    The task of identifying inputs that differ significantly from the patterns of normal data, often without any labeled examples of the anomalies themselves (e.g., flagging never-before-seen network intrusions by their high {term}`Reconstruction Error`).

API (Application Programming Interface)
    A set of protocols that allows different software applications to communicate. In AI, APIs are used to send data (like raw text) to powerful cloud-hosted models (like LLMs) and securely receive the processed output.

API Key
    A secret string that identifies a program to a web service and is used for authentication and billing. Should be stored in an environment variable or secrets manager, never hard-coded or committed to version control.

Approximate Nearest Neighbor (ANN) Search
    A family of indexing methods (such as HNSW graphs) that find vectors very close to a query vector by examining only a small fraction of the collection. Much faster than comparing against every vector, at the cost of occasionally missing a true top match.

Artificial Intelligence (AI)
    The study of how to make computers do things at which, at the moment, people are better (e.g., rapid pattern recognition, natural language processing, and complex decision-making).

Artificial Neural Network (ANN)
    A computing system inspired by biological brains, consisting of interconnected nodes (neurons) organized in layers that automatically learn complex, non-linear representations of raw data.

Autoencoder
    An unsupervised neural network architecture consisting of an encoder and decoder, trained to reconstruct its own input. Used for anomaly detection (such as zero-day cyber intrusions) by measuring reconstruction failure, though unusual inputs are not necessarily malicious.

Autoregressive Generation
    Producing output one token at a time, where each newly generated token is appended to the input and used to predict the next one. The way decoder-only LLMs write their replies.

Backpropagation
    The core algorithm used to train neural networks. It calculates the gradient (error) of the loss function with respect to every weight in the network, mathematically working backward from the output layer to the input layer.

Bagging (Bootstrap Aggregating)
    An ensemble machine learning technique that trains multiple models (like Decision Trees) on random subsets of data and averages their predictions to reduce variance and prevent overfitting.

Base Model
    A foundational Large Language Model that has completed pre-training but has not yet undergone instruct-tuning. It excels at predicting the next word and understanding language, but cannot reliably follow commands or act as an assistant.

Base Rate
    The underlying, prior probability of an event or condition occurring in a given population or environment before any specific new evidence is considered.

Batch
    A subset of training examples processed together in one forward and backward pass, producing a single weight update. Also called a mini-batch.

Batch Normalization
    A layer that rescales the outputs of the previous layer during training so they stay in a consistent range, stabilizing and speeding up the training of deep networks.

Bayes' Rule
    A mathematical theorem describing how to update the probability of a hypothesis based on new evidence, relating the posterior probability to the prior probability, likelihood, and marginal likelihood.

Bayesian Network
    A probabilistic graphical model that represents a set of random variables and their conditional dependencies via a directed acyclic graph (DAG). It decomposes a full joint probability distribution into local conditional probability distributions.

Belief State (Filtering)
    The probability distribution over the current state of a dynamical system conditioned on the entire history of observed emissions up to that time step: $B(X_t) = P(X_t \mid e_{1:t})$.

Bias (Neural Networks)
    A learnable constant ($b$) added to a neuron's weighted sum. It acts as a baseline threshold, shifting the activation function left or right so the neuron can fire even if all input features are zero.

Bias (Statistical Error)
    The error introduced when a model makes overly simplistic assumptions about the data (e.g., assuming a relationship is a straight line when it is actually a curve). High bias leads to {term}`Underfitting (High Bias)`.

Bias-Variance Tradeoff
    The fundamental tension in machine learning where decreasing a model's false assumptions ({term}`Bias (Statistical Error)`) typically increases its dangerous sensitivity to noise ({term}`Variance (Statistical Error)`), and vice versa.

Branching Factor
    The number of successor actions available to an agent from a given node in a search tree, heavily influencing the time complexity of a search algorithm.

Breadth-First Search (BFS)
    An uninformed search algorithm that explores the state space identically in all directions, layer-by-layer. It utilizes a FIFO queue, ensuring the shallowest nodes are expanded first.

Byte-Pair Encoding (BPE)
    A tokenizer-training algorithm that starts from single characters (or bytes) and repeatedly merges the most frequent adjacent pair of symbols into a new vocabulary entry. Frequent words become single tokens, while rare or unseen words are split into smaller pieces.

Cascading Hallucination
    A failure mode unique to multi-agent systems where one agent hallucinates a fact or capability, and other agents subsequently accept that hallucination as truth, rapidly derailing the entire workflow or simulation.

Catastrophic Forgetting
    The loss of useful pre-trained knowledge when a network is trained on new data with too large a learning rate, so that its existing weights are quickly overwritten. Avoided during fine-tuning by freezing early layers and using a small learning rate.

Causal Inference
    Probabilistic inference that flows in the direction of causal edges (from parent causes to child effects), evaluating $P(\text{Effect} \mid \text{Cause})$.

Causal Masking
    Blocking each token from attending to later tokens by setting those attention scores to negative infinity before the softmax. Used in generating models so that next-token prediction cannot see the answer.

Chain-of-Thought (CoT) Prompting
    Prompting an LLM to write out intermediate reasoning steps before its final answer, either by including worked examples with reasoning or by adding an instruction such as "Let's think step by step." Often improves accuracy on multi-step problems at the cost of more output tokens.

Chat Template
    The fixed format, using special marker tokens, that converts a list of system, user, and assistant messages into the single token sequence an instruct model actually reads. Matches the format used during instruction tuning.

Child (Bayesian Network)
    In a directed graph, a node that has an incoming directed edge pointing to it from another node (its parent). If $A \to B$, then $B$ is the child of $A$.

Chunk Overlap
    The number of tokens repeated between the end of one chunk and the start of the next, so that sentences near a boundary appear intact in at least one chunk. The step between chunk starts (the stride) equals the chunk size minus the overlap.

Classification Head
    The final layer (or layers) of a neural network that maps learned features to the output classes. In transfer learning, the original head is removed and replaced with a new, randomly initialized head sized for the new task.

Completeness
    A property of an AI search algorithm that indicates whether the algorithm is mathematically guaranteed to eventually find a solution if one exists within the state space.

Conditional Independence
    A condition where two variables or events, $A$ and $B$, are independent given the knowledge or observation of a third conditioning variable $C$, satisfying $P(A, B \mid C) = P(A \mid C)P(B \mid C)$ or $P(A \mid B, C) = P(A \mid C)$.

Conditional Probability
    The mathematical probability of an event $A$ occurring given that another conditioning event $B$ is already known to have occurred, defined as $P(A \mid B) = \frac{P(A, B)}{P(B)}$ (where $P(B) > 0$).

Conditional Probability Table (CPT)
    A table associated with a node in a discrete Bayesian network that specifies the conditional probability distribution of that node given every possible combination of values taken by its parent nodes.

Confusion Matrix
    A performance measurement table that visualizes exactly how a classification model succeeds or fails, categorizing predictions into True Positives, True Negatives, False Positives, and False Negatives. Crucial for weighing operational risks.

Connectionism
    A subdiscipline of AI popularized in the 1980s that abandoned explicit, programmed rules in favor of artificial neural networks that learn from training examples.

Consistency
    A property of a heuristic where the estimated cost to the goal from node A is no greater than the step cost to a neighboring node C, plus the estimated cost from C to the goal ($h(A) \le \text{cost}(A \to C) + h(C)$). If consistent, A* Graph Search is guaranteed to be optimal.

Context Window
    The maximum number of tokens (input plus generated output) a Large Language Model can process in a single request. Because API calls are stateless, the whole conversation history must fit inside it. In agentic AI, this serves as the agent's Short-Term Memory.

Convolution
    A mathematical operation where a small matrix (filter/kernel) slides across an input image, multiplying and summing pixel values to detect specific visual features like edges, shapes, or textures.

Convolutional Neural Network (CNN)
    A specialized deep learning architecture designed to process grid-like data (such as images). It uses sliding filters to learn spatial hierarchies of features, making it highly effective for visual ISR and target recognition tasks.

Cosine Similarity
    A measure of how similar two vectors are, computed as their dot product divided by the product of their lengths, which equals the cosine of the angle between them. Ranges from -1 (opposite) through 0 (unrelated) to 1 (same direction) and ignores vector length. In semantic search, the ranking of scores matters more than any fixed cutoff; for normalized vectors it equals the dot product.

Cost Matrix
    A table assigning a cost to each type of prediction outcome (for example, the cost of a false negative versus a false positive). Multiplying each cell of a confusion matrix by its cost and summing gives the expected cost of a model or threshold.

Cross-Correlation
    The sliding weighted-sum operation that deep-learning libraries actually compute in a "convolutional" layer. It is identical to mathematical convolution except that the filter is not flipped; because filters are learned, the difference has no practical effect.

Cross-Entropy Loss
    The standard loss function for classifiers that output probabilities. It heavily penalizes confident wrong predictions: $L = -[y \ln p + (1-y) \ln(1-p)]$.

Data Augmentation
    Creating additional training examples by applying label-preserving transformations (such as rotations, flips, or brightness changes) to existing data. Reduces overfitting when labeled data is limited; valid transformations depend on the domain (all 90-degree rotations are valid for overhead imagery but not for street-level photos).

Data Poisoning
    A training-phase attack where an adversary infiltrates a data pipeline and subtly alters the training examples or labels, causing the AI to learn a malicious, hidden rule (a "backdoor") that the adversary can exploit later in combat.

Decision Threshold
    The probability cutoff that converts a classifier's output score into a yes-or-no prediction. Lowering it raises recall and lowers precision; raising it does the opposite. Choosing it is where mission risk tolerance enters the system.

Decision Tree
    A supervised learning model that classifies data by recursively splitting it into branches based on the feature that best purifies the resulting groups (often measured by {term}`Gini Impurity`).

Decoder
    The second half of an Autoencoder that attempts to decompress the latent space bottleneck back into the original data format.

Decoder-Only Transformer
    A Transformer that uses causal masking so each token sees only earlier tokens, trained to predict the next token. The architecture of most generative LLMs, including the GPT, Claude, and Llama families. Compare encoder-only models such as BERT, which let every token see the whole input.

Decoding (HMM)
    The inferential task of determining the single most likely sequence of hidden states that generated an observed sequence of emissions: $\arg\max_{x_{1:t}} P(x_{1:t} \mid e_{1:t})$.

Deep Learning
    A subset of machine learning utilizing neural networks with many hidden layers (deep architectures) capable of autonomously engineering complex, hierarchical features from raw, unstructured data (like images or RF signals).

Delimiters
    Markers such as XML-style tags, triple quotes, or headings that separate instructions from data (documents, code, user input) inside a prompt. They help the model distinguish the two and reduce, but do not eliminate, the risk of prompt injection.

Depth-Limited Search
    A variation of Depth-First Search that imposes a strict maximum depth limit to prevent the algorithm from plunging down infinite paths, forcing it to backtrack once the limit is reached.

Depthwise Separable Convolution
    A lightweight replacement for a standard convolution that splits it into a depthwise step (one spatial filter per input channel) and a pointwise step (a 1 × 1 convolution that mixes channels). Uses far fewer parameters and operations; the basis of MobileNet architectures.

Descendant (Bayesian Network)
    In a directed graph, any node that can be reached from a given node by following a path of directed edges downstream.

Deterministic Policy
    A policy that maps each specific state to exactly one single, definitive action.

Diagnostic Inference
    Probabilistic inference that flows against the direction of causal edges (from observed symptoms/effects back to hidden causes), evaluating $P(\text{Cause} \mid \text{Effect})$.

Directed Acyclic Graph (DAG)
    A finite directed graph containing vertices and directed edges that has no directed cycles. That is, starting at any vertex and following the directed edges, one can never loop back to the same vertex.

Discount Factor
    A mathematical multiplier ($\gamma$) bounded between 0 and 1 used in reinforcement learning and MDPs to value near-term rewards more heavily than distant future rewards, ensuring that infinite operational loops converge to a finite mathematical value.

Distribution Shift (Domain Shift)
    A mismatch between the data a model was trained on and the data it encounters in deployment (e.g., daytime training imagery versus nighttime operations), which can sharply degrade performance even when test accuracy was high.

Document Chunking
    Splitting documents into smaller passages (chunks) so each can be embedded, searched, and inserted into an LLM prompt separately. Chunk size trades retrieval precision (small chunks) against surrounding context (large chunks); chunks often overlap and follow the document's structure.

Dominance
    A method of comparing heuristic quality. If heuristic A is mathematically greater than or equal to heuristic B for every node in the state space ($h_a(n) \ge h_b(n)$), then A dominates B. A dominant heuristic guarantees A* explores fewer nodes.

Dropout
    A regularization technique where a random percentage of neurons in a layer are temporarily disabled during each training step. This prevents the network from relying on specific neurons to memorize noise, forcing it to learn robust, generalized features.

Early Stopping
    A regularization technique that monitors the validation loss during training. It automatically halts the training process when the model stops improving on unseen data, saving the optimal weights before overfitting occurs.

Edge Device
    A lightweight, low-power hardware payload (like a drone's onboard computer) capable of processing neural network inference locally without requiring a connection to a centralized cloud server.

Embedding Model
    A neural network that converts a piece of text (or an image) into a fixed-length vector whose direction captures its meaning, so that similar texts produce similar vectors. Queries and documents must be embedded with the same model to be comparable.

Emission Model (Sensor Model)
    In a Hidden Markov Model, the conditional probability distribution defining the likelihood of observing a particular evidence symbol given the true hidden state: $P(E_t \mid X_t)$.

Encoder
    The first half of an Autoencoder that compresses raw input data down into a mathematically dense bottleneck.

Epoch
    One complete pass of the entire training dataset through the neural network during the training phase.

Evidence Variable
    A variable in a probabilistic model whose value has been directly observed or measured by sensors ($E = e$).

Expected Return ($G$)
    The cumulative sum of all discounted future rewards an agent expects to accumulate starting from a given time step.

Expected Value (EV)
    The mathematical average of all possible outcomes of a chance event, calculated by multiplying each possible outcome by its probability of occurrence and summing the results.

Expectimax
    A decision-making algorithm used in stochastic environments. Instead of assuming an opponent will always make the worst possible move for the agent, it calculates the mathematically expected utility of chance events.

Expert System
    An early form of Symbolic AI that attempts to mimic human expertise by utilizing an inference engine to process a massive knowledge base of hardcoded, explicit "if-then" rules.

Explainable AI (XAI)
    A field of research and a set of mathematical tools (like SHAP) designed to make the internal mechanics and outputs of "black box" machine learning models understandable to human operators, ensuring trust and traceability.

Exploding Gradient
    A training failure where gradients grow uncontrollably large as they are multiplied backward through many layers, causing unstable weight updates and a loss that diverges (often to NaN).

Exponential Backoff
    A retry strategy in which a client waits progressively longer between attempts (for example 1, 2, 4, then 8 seconds) after a temporary failure such as a rate-limit error.

F-beta Score
    A generalization of the F1-score that weights recall $\beta$ times as heavily as precision. $F_2$ favors recall (missing a target is worse); $F_{0.5}$ favors precision (false alarms are worse).

F1-Score
    The harmonic mean of Precision and Recall. It provides a single, balanced mathematical metric to evaluate a model's performance, especially when dealing with highly imbalanced datasets.

Faithfulness
    In RAG evaluation, the degree to which every claim in a generated answer is supported by the retrieved sources it cites. An answer can be fluent and even correct but unfaithful if it relies on information not in the sources.

False Negative (Type II Error)
    A failure where the model predicts that a condition is absent, but it is actually present (e.g., the AI predicts "no threat," but an enemy is there). In warfare, this is often the most fatal type of error.

False Positive (Type I Error)
    A failure where the model predicts that a condition is present, but it is actually absent (e.g., the AI predicts an enemy, but it is a civilian). This leads to false alarms, wasted resources, and collateral damage.

Feature Engineering
    The manual process of designing and computing input features from raw data using human domain expertise (e.g., measuring pulse width from a radar signal). Deep learning largely replaces this step with learned representations.

Feature Extraction
    The process of isolating the most relevant mathematical variables from preprocessed data to feed into an AI model.

Feature Extraction (Transfer Learning)
    A transfer learning strategy in which the entire pre-trained base network is frozen and used only to compute features, while a new {term}`Classification Head` is trained on top.

Feature Importance
    A technique used to interpret machine learning models by calculating and ranking exactly which input variables (features) had the greatest mathematical impact on the model's final predictions.

Feature Map
    The output image or matrix produced after a convolution filter has scanned across the input data, mathematically highlighting exactly where specific visual features (like lines or shapes) are located.

Feedforward Architecture
    A neural network design where information moves in only one direction (from the input layer, through hidden layers, directly to the output layer) without looping back.

Few-Shot Prompting
    Including a few examples of the desired input-output behavior in the prompt so the model can infer the task and output format without retraining. The model tends to copy patterns in the examples, including accidental ones, so examples should be varied.

FIFO (First-In, First-Out) Queue
    A data structure where the oldest added item is the first one removed. It is used to manage the frontier in Breadth-First Search, causing the algorithm to explore in shallow tiers.

Filter (Kernel)
    A small grid of learnable weights used in a CNN that slides across an image to detect specific patterns, such as horizontal lines or color gradients.

Filtering (Monitoring)
    The process of computing the belief distribution over the current hidden state given all historical observations up to the present moment: $P(X_t \mid e_{1:t})$.

Fine-Tuning
    A form of transfer learning in which the upper layers of a pre-trained network are "unfrozen" and trained on domain-specific data with a much smaller learning rate, usually after a new classification head has first been trained with the base frozen. Used to adapt a model to a different domain, such as shifting from ground-level photographs to top-down satellite imagery.

FLOPs and MACs
    Measures of the arithmetic a model performs for one input. FLOPs counts floating-point operations; MACs counts multiply-accumulate operations (one multiplication plus one addition). Usually reported in billions, they approximate a model's compute time and energy use.

Follow-On Effects
    The downstream, cascading consequences of an agent's action in a sequential environment, often extending beyond the immediate state transition.

Forward Algorithm (Filtering)
    A dynamic programming algorithm for HMMs that recursively computes the filtered belief state by alternating between a time prediction step and an observation update step.

Forward Propagation
    The process of pushing raw input data forward through the hidden layers of a neural network to generate a final prediction. It involves calculating the weighted sums and applying activation functions at each neuron in the sequence.

Foundation Model
    A very large neural network pre-trained on broad data at massive scale and designed to be adapted, through fine-tuning or prompting, to many downstream tasks. Large Language Models are foundation models.

Freezing (Frozen Layers)
    Marking a layer's weights as non-trainable so that gradient descent cannot change them. Used in transfer learning to protect pre-trained features from being overwritten by noisy gradients from a new, untrained head.

Fringe
    In a search algorithm, the frontier or collection of unexpanded nodes waiting to be explored.

Frontier
    (Also known as the Fringe). The collection of all unexplored nodes that the AI agent has discovered but not yet expanded. How this data structure is managed dictates the search algorithm's tactical behavior.

Generalization
    The ultimate goal of a machine learning model: the ability to learn the true underlying patterns of a dataset so successfully that it performs highly accurately on brand new, unseen data.

Gini Impurity
    A mathematical metric used by Decision Trees to measure how "mixed" or impure a node of data is. A score of 0 means perfect purity (the node contains only one class).

Global Average Pooling
    A layer that averages each final feature map down to a single number, turning an H × W × C output into C values. Commonly replaces flatten-plus-dense layers at the end of a CNN, greatly reducing parameters, discarding position, and enabling class activation heatmaps.

Goal State
    The target configuration or end condition that a rational agent is attempting to reach in a search problem.

Grad-CAM (Gradient-weighted Class Activation Mapping)
    A technique that produces a heatmap showing which regions of an image most increased a CNN's score for a given class, computed from the gradients of that score with respect to the last convolutional layer's feature maps. Provides evidence, not proof, of what the model relied on.

Gradient
    A mathematical vector of partial derivatives representing the direction of steepest ascent for the loss function.

Gradient Clipping
    A technique that caps the size of gradients (or weight updates) during training to prevent {term}`Exploding Gradient`s from destabilizing the network.

Gradient Descent
    The optimization algorithm used to minimize a network's loss. It updates the network's weights iteratively by taking small mathematical steps in the opposite direction of the {term}`Gradient`.

Greedy
    An informed search algorithm that expands the node that appears closest to the goal based entirely on the heuristic function ($h(n)$), ignoring the cost of the path taken so far. It is fast but heavily prone to suboptimal paths and getting trapped by obstacles.

Ground Sample Distance (GSD)
    The width of ground represented by one pixel in overhead imagery (for example, 0.3 m or 10 m). Determines how many pixels an object occupies and therefore what a model can detect.

Grounding
    Instructing an LLM to base its response only on provided source material (such as retrieved documents), to cite those sources, and to say when they do not contain the answer, rather than relying on its pre-trained knowledge.

Hallucinations
    A phenomenon where a Large Language Model (LLM) or generative AI generates false, fabricated, or nonsensical information but presents it with high confidence as if it were a verified fact. This occurs because the model is predicting statistically likely word sequences, not retrieving verified truth.

Hardware Agent (Autonomous Robot)
    An agent equipped with physical sensors (to perceive the environment) and physical actuators (to manipulate the environment).

Heuristic
    A mathematical, educated guess used to estimate the remaining cost from a current state to the goal state. It guides an AI's search process to explore promising paths first, vastly reducing the mathematical search space.

Hidden Layer
    A layer of artificial neurons situated between the input and output layers of a neural network. These layers are responsible for learning abstract, hidden features in the data.

Hidden Markov Model (HMM)
    A temporal probabilistic model where the underlying system state evolves according to a Markov process but is unobservable (hidden), generating noisy observable emissions at each time step.

Hidden State
    An unobserved internal state variable ($X_t$) in an HMM that cannot be directly perceived, requiring probabilistic inference from observable emissions.

Hidden Variable
    A variable in a probabilistic system that is neither an explicit query nor an observed piece of evidence, which must be summed out (marginalized) during inference.

Hierarchical Feature Learning
    The process by which deep neural networks autonomously learn simple concepts (like lines and edges) in early layers and mathematically combine them into complex tactical concepts (like vehicles or radar structures) in deeper layers.

Hybrid Search
    Combining semantic (vector) search with keyword search and merging their results, so a system finds passages that match a query's meaning or its exact terms, such as error codes, part numbers, and names.

Hyperplane
    The mathematical decision boundary drawn by a Support Vector Machine to separate classes. In 2D space, it is a line; in 3D space, it is a flat plane.

Independence (Statistical Independence)
    A condition where the occurrence or observation of one event provides no mathematical information about the probability of another event, satisfying $P(A, B) = P(A)P(B)$ and $P(A \mid B) = P(A)$.

Indirect Prompt Injection
    A prompt injection in which malicious instructions are hidden in content the model reads, such as a retrieved document, web page, or email, rather than typed by the user. A key risk for RAG systems and AI agents.

Inference
    The phase in an AI pipeline where a trained model or logic system applies its learned rules to new, unseen data to generate a prediction or decision.

Inference by Enumeration
    An exact algorithm for answering probabilistic queries in a Bayesian network by expanding the joint distribution via graph factorization and summing out all hidden variables.

Inference Latency
    The time it takes for a deployed neural network to process a single input (like a video frame) and output a prediction. Critical for autonomous edge devices tracking fast-moving targets.

Inference Mechanism
    The processing engine or program in a knowledge-based system that parses the declarative rules stored in the Knowledge Base to draw conclusions or answer queries.

Informed Search
    A class of search algorithms that utilizes domain-specific knowledge (heuristics) to find solutions more efficiently than blind (uninformed) search methods.

Instruct-Tuning
    The post-training stages that turn a base model into an assistant: supervised fine-tuning on example prompt/response pairs, usually followed by preference tuning such as RLHF. It makes a model follow instructions far more reliably, but does not make it always correct or immune to manipulation.

Irreducible Error
    The inherent noise, randomness, or "fog of war" in a dataset that no machine learning model can ever predict or eliminate.

Iteration
    One trip through the training loop on a single batch: forward pass, loss, backpropagation, and one weight update. The number of iterations per epoch equals the number of training examples divided by the batch size.

Iterative Prompting
    Refining an LLM's output over several turns by treating each response as a draft and giving targeted follow-up instructions. Works because the conversation history is resent with each request.

Joint Probability
    The probability of two or more random variables or events occurring simultaneously, denoted as $P(A, B)$ or $P(A \cap B)$.

Kernel Trick
    A mathematical technique used by Support Vector Machines (SVMs) to project non-linear, complex data into higher dimensions where it can be easily sliced by a flat hyperplane.

Keyword Search (BM25)
    Ranking documents by how often the query's words appear in them, weighted by how rare those words are across the collection; BM25 is the standard formula. Excels at exact matches that embeddings can blur.

Knowledge Base (KB)
    A centralized repository of declarative rules, logic, or data that is explicitly separated from the system's processing/inference mechanism, allowing the knowledge to be updated independently.

Knowledge Distillation
    A model-compression technique in which a small "student" network is trained to imitate the outputs of a large, accurate "teacher" network, often achieving better accuracy than the same small network trained alone.

L2 Regularization (Weight Decay)
    A regularization technique that adds a penalty proportional to the sum of the squared weights ($\lambda \sum w^2$) to the loss, pushing the network toward smaller weights and smoother functions that generalize better. A very large $\lambda$ causes underfitting.

Laplace Smoothing (Add-One Smoothing)
    A mathematical technique used in Naive Bayes to prevent the formula from collapsing to zero when the model encounters a feature (like a new vocabulary word) it has never seen before.

Large Language Model (LLM)
    A massive deep learning model based on the Transformer architecture, trained on enormous datasets to understand, generate, and reason over human language and unstructured text.

Latency Budget
    The maximum processing time allowed per input for a system to keep up with its data stream. For a 30-frames-per-second camera, the latency budget is about 33 milliseconds per frame.

Latent Space (Bottleneck)
    The highly compressed, mathematical representation of data found in the middle layer connecting an encoder and decoder within an Autoencoder.

Layer Normalization
    A normalization step that rescales the values within each token's vector to a steady mean and spread. Used inside every Transformer block to stabilize training; similar in purpose to batch normalization but computed per example.

Learning
    An AI methodology where a system automatically identifies patterns and rules from historical training data, rather than being explicitly programmed.

Learning Rate
    A hyperparameter ($\alpha$) that dictates how large of a step the network takes during {term}`Gradient Descent`. If it is too large, the network overcorrects erratically; if it is too small, learning is painfully slow.

LIFO (Last-In, First-Out) Stack
    A data structure where the most recently added item is the first one removed. It is used to manage the frontier in Depth-First Search, causing the algorithm to plunge as deeply as possible down a single path.

Likelihood
    In Bayes' Theorem, the probability of observing a specific piece of evidence assuming that a particular hypothesis is true.

Local Receptive Field
    The small region of the input (e.g., a 3 × 3 patch of pixels) that a single neuron in a convolutional layer connects to.

Logistic Regression
    A foundational classification algorithm that passes a linear equation through a {term}`Sigmoid Function` to predict the probability of a binary outcome.

Logits
    The raw, unnormalized scores a network outputs for each class before softmax. In an LLM, the final layer produces one logit for every token in the vocabulary.

Long Short-Term Memory (LSTM)
    A type of {term}`Recurrent Neural Network (RNN)` that uses gating mechanisms to decide what information to remember and what to forget, allowing it to learn longer-range patterns in sequences than a plain RNN.

Long-Term Memory (Agentic)
    An AI agent's ability to store and recall information across different, isolated chat sessions, typically implemented using a Vector Database and Retrieval-Augmented Generation (RAG).

Loss Function (Cost Function)
    A mathematical function that evaluates how far off a network's predictions are from the true targets. The network's entire training goal is to minimize this value.

Marginal Probability (Marginalization)
    The unconditional probability of an event obtained by summing (or integrating) the joint probability across all possible outcomes of all other random variables: $P(A) = \sum_b P(A, B=b)$.

Markov Chain
    A discrete-time stochastic process that satisfies the Markov property, where the probability of transitioning to the next state depends solely on the current state.

Markov Decision Process (MDP)
    A mathematical framework for modeling decision-making in stochastic environments where outcomes are partly random and partly under the control of a decision-maker. It is defined by a set of states, actions, transition probabilities, and reward functions.

Markov Property
    The foundational assumption in temporal and sequential models that the future state depends strictly and solely on the current state and the current action taken, and is completely independent of the past sequence of events that led to the current state.

Max Pooling
    A downsampling operation commonly used in CNNs that slides a window across a feature map, keeping only the most prominent signal (the maximum value) while discarding the rest. This reduces computational load and provides translation invariance.

Mean Squared Error (MSE)
    A common loss function used for regression tasks (predicting continuous numbers) that calculates the average squared difference between the predicted values and the actual target values.

Meta-Prompt
    An instruction about how the model should work rather than what it should produce, such as "Break this problem into parts before answering" or "Ask me clarifying questions before you begin."

Metadata Filtering
    Restricting a search to chunks whose stored attributes (source, date, version, access level) meet given conditions, for example only documents the current user is authorized to see. Enforcing access control at this step keeps unauthorized text out of the LLM's prompt.

Minimax
    A foundational adversarial decision-making algorithm where one agent (MAX) attempts to maximize a tactical score, while the opposing agent (MIN) acts perfectly to minimize that same score.

Multi-Head Attention
    Running several self-attention calculations in parallel, each with its own learned query, key, and value matrices, and concatenating the results. Lets the model capture several kinds of relationships between tokens at once.

Multilayer Perceptron (MLP)
    A feedforward neural network in which every neuron in one layer connects to every neuron in the next (fully connected layers). Well suited to tabular data, but inefficient for raw images, audio, or text.

Nadir Perspective
    A viewpoint looking straight down at the ground, as in most satellite and high-altitude drone imagery. Objects seen from nadir look very different from the ground-level photographs most pre-trained models learned from.

Naive Assumption (Independence Assumption)
    The core (and mathematically flawed) assumption in Naive Bayes that every feature in a dataset is completely independent of every other feature.

Next-Token Prediction
    The training objective of generative LLMs: given the tokens so far, output a probability distribution over the vocabulary for the next token. A classification problem trained with cross-entropy loss.

Node
    A discrete point or data structure. In a state space graph, it represents a single physical configuration. In a search tree, it represents an entire path or plan taken from the start state. In a Bayesian network, it represents a random variable.

Normalization Constant ($\alpha$)
    A scaling factor (equal to $\frac{1}{\sum P}$) multiplied by an unnormalized joint probability distribution to guarantee that the resulting posterior probabilities sum to exactly 1.0.

Observation (Emission)
    The measurable signal, sensor reading, or feature value generated by a physical system at time step $t$ according to its emission model.

Observation (ReAct)
    The step in the ReAct loop where the result of an external tool call is returned and injected back into the agent's context window, allowing the agent to evaluate the outcome of its action.

Open-Weight Model
    A model whose trained weights are published so it can be downloaded and run on one's own hardware (for example, Llama). Useful when data cannot be sent to an external API.

Optimal State-Value Function
    The maximum expected discounted return achievable from each state, assuming the agent takes the best possible actions from that state onward until termination.

Optimality
    The guarantee that an AI search algorithm will find the absolute lowest-cost path to a goal state, assuming such a path exists.

Overfitting
    A failure state (driven by high variance) where a model becomes so overly complex that it memorizes the random noise in its training data rather than learning the underlying tactical rules. It performs perfectly in training but fails on unseen data.

Padding (Zero-Padding)
    The technique of adding a border of zero-value pixels around the edges of an input image before applying a convolution. This prevents the resulting feature map from shrinking and preserves edge information.

Parent (Bayesian Network)
    In a Bayesian network graph, a node that has an outgoing directed edge directly pointing to another node (its child). If $A \to B$, then $A$ is an immediate parent of $B$.

Path Cost
    A numerical value used in search problems to compare the effects (e.g., fuel, time, risk) of different possible sequences of actions.

Perceptron
    The simplest historical form of an artificial neuron. It takes multiple inputs, calculates a weighted sum, adds a bias, and passes the result through a hard step-function to output a 1 or a 0.

Pipeline
    The highly structured sequence of operations in an AI system that dictates how raw input data is ingested, processed, modeled, and translated into an actionable output.

Policy (MDP)
    A comprehensive strategy or mapping that dictates exactly what action an agent should take from any given state in its environment.

Positional Encoding
    Information added to each token's embedding that indicates its position in the sequence, so a Transformer processing all tokens in parallel can still tell word order. Variants include fixed sinusoidal encodings, learned position vectors, and rotary position embeddings (RoPE).

Posterior Probability
    In Bayes' Theorem, the final, updated probability of a hypothesis being true *after* factoring in the new evidence.

Precision
    An operational metric answering: "Out of all the times the model predicted a positive target, how many were actually correct?" Calculated as TP / (TP + FP).

Precision-Recall (PR) Curve
    A graph that plots Precision against Recall across different probability thresholds. It is highly valuable for evaluating models trained on severely imbalanced datasets, visually demonstrating the tactical tradeoff between catching all threats and avoiding false alarms.

Preprocessing
    The step immediately following data ingestion where raw data is cleaned, normalized, or filtered to make it mathematically usable for the model.

Pre-Training
    The initial, most expensive training phase of an LLM, in which it learns from a very large body of unlabeled text (often trillions of tokens) by repeatedly predicting the next token. Produces a base model that has absorbed grammar, facts, and patterns but is not yet an assistant.

Prior Probability
    In Bayes' Theorem, the baseline, historical probability of a hypothesis occurring *before* any new evidence is observed.

Priority Queue
    A data structure that manages nodes based on a specific numerical priority rather than the order they were inserted. In Uniform Cost Search, nodes are prioritized strictly by the lowest cumulative path cost.

Probability Distribution
    A mathematical function or table that assigns a probability to every possible mutually exclusive outcome or state in an entire sample space, such that all assigned probabilities are non-negative and sum to exactly 1.0.

Product Rule (Probability)
    The mathematical rule stating that the joint probability of two events can be factored into a conditional probability multiplied by a marginal probability: $P(A, B) = P(A \mid B)P(B) = P(B \mid A)P(A)$.

Prompt Chaining
    Breaking a complex task into a sequence of smaller prompts, where each step's output becomes the next step's input, often with ordinary code between steps. Easier to test and debug than one large prompt.

Prompt Engineering
    The practice of designing, structuring, testing, and refining the text sent to a Large Language Model so that it reliably produces accurate, specific, and useful output. Includes defining the task, format, voice, and context; giving examples; requesting step-by-step reasoning; and evaluating prompts on test sets like code.

Prompt Evaluation
    Measuring a prompt's performance by running it on a fixed test set of realistic inputs with known expected outputs and scoring the results. Used to compare prompt versions and detect regressions when a prompt or model changes; often called an eval.

Prompt Injection
    A failure mode or cyber attack where malicious input is designed to bypass an LLM's safety filters, causing it to ignore its original system instructions and execute an unauthorized command.

Prompt Template
    A reusable prompt containing placeholders (for example, {topic} or {document}) that a program fills in for each request, so the same tested instructions are applied consistently to many inputs.

Pruning
    A model-compression technique that removes weights or entire filters contributing little to the output, followed by brief retraining to recover accuracy.

Quantization
    Storing a model's weights (and often performing its arithmetic) at lower numerical precision, typically 8-bit integers instead of 32-bit floating-point numbers. Reduces model size about 4 times and speeds up inference on supported hardware, usually with a small accuracy loss that must be re-measured.

Query Variable
    A variable in a probabilistic graphical model whose posterior probability distribution is to be calculated given observed evidence ($Q$).

Query, Key, and Value (Attention)
    The three vectors computed for each token in attention by multiplying its embedding by learned weight matrices W_Q, W_K, and W_V. The query is what a token is looking for, the key is what it can be matched on, and the value is the information it passes along. Attention works like a soft dictionary lookup.

Random Forest
    A powerful ensemble model that builds hundreds of Decision Trees, each trained on a bootstrap sample of the data and limited to a random subset of features at each split, then averages (or votes over) their predictions. The individual trees are typically grown deep; averaging many decorrelated trees reduces variance while keeping bias low.

Rate Limit
    A cap a service places on how many requests or tokens a client may use per unit of time. Exceeding it typically returns an HTTP 429 error, which should be handled by retrying with exponential backoff.

ReAct (Reasoning and Acting)
    An agentic architecture and prompting strategy that forces an LLM to alternate between articulating explicit reasoning (Thought) and executing tool calls (Action), using the results (Observation) to guide subsequent steps.

Recall (Sensitivity)
    An operational metric answering: "Out of all the true positive targets that actually existed in the environment, how many did the model successfully find?" Calculated as TP / (TP + FN).

Recall@k
    A retrieval metric: the fraction of test questions for which a correct passage appears among the top k retrieved results. Used to choose k, chunk size, and search method in a RAG system.

Receptive Field
    The region of the original input that can influence a given neuron's value. It grows with each stacked convolutional layer and grows faster after pooling or strided layers, which is how deep CNNs build a spatial hierarchy from edges to whole objects.

Reconstruction Error
    The mathematical difference (often measured via MSE) between an original input and an Autoencoder's attempted reconstruction of it. Spikes in this error trigger anomaly alerts.

Recurrent Neural Network (RNN)
    A neural network architecture for sequential data that processes one element at a time while carrying a hidden memory vector forward from step to step. Largely superseded by Transformers for large-scale language tasks.

Reflex Agent
    An AI agent that makes decisions purely by reacting to current, immediate input/perceptions, without relying on internal memory of past events.

Regularization
    A set of techniques (like {term}`Dropout`, {term}`Early Stopping`, or {term}`L2 Regularization (Weight Decay)`) used to constrain or penalize complex models to prevent them from overfitting the training data.

Reinforcement Learning from Human Feedback (RLHF)
    A preference-tuning method in which people compare pairs of model responses, a reward model learns to predict their preferences, and the LLM is then trained with reinforcement learning to produce responses the reward model rates highly.

ReLU (Rectified Linear Unit)
    A highly efficient and popular activation function used in deep learning hidden layers. It outputs the raw input directly if it is positive, and outputs zero if it is negative.

Representation Learning
    The ability of a model to automatically discover the features (representations) needed for a task directly from raw data, rather than relying on hand-engineered features. The core idea behind deep learning.

Residual Connection (Skip Connection)
    A shortcut that adds a block's input directly to its output ($\text{output} = F(x) + x$), giving gradients a direct path backward through the network. Residual connections made networks with 100 or more layers trainable (ResNets) and appear in nearly all modern deep architectures.

Retrieval-Augmented Generation (RAG)
    An architecture in which a system first retrieves the passages most relevant to a question from an external collection of documents, then inserts them into the LLM's prompt with instructions to answer from them and cite them. Gives the model access to current or private information without retraining and reduces, but does not eliminate, hallucination.

Reward Function
    The immediate numerical feedback $R(s,a,s')$ an agent receives from the environment after transitioning from state $s$ to state $s'$ due to executing action $a$.

ROC Curve (Receiver Operating Characteristic)
    A graph showing the performance of a classification model at all classification thresholds. It plots the True Positive Rate against the False Positive Rate. The area under this curve (AUC) provides an aggregate measure of performance across all possible thresholds.

Role Prompting
    Assigning the model a role or persona (for example, "You are a senior security engineer"), usually in the system prompt, to shape its style, focus, and level of detail. A role does not add knowledge the model lacks.

Rotary Position Embedding (RoPE)
    A positional-encoding method that rotates each token's query and key vectors by angles that depend on its position, so attention scores reflect the relative distance between tokens. Used in many recent open models, including Llama.

Scaled Dot-Product Attention
    The standard attention formula, softmax(QK^T / sqrt(d_k))V. Dividing by the square root of the key dimension keeps the scores from growing so large that the softmax becomes overly peaked.

Search
    A problem-solving AI technique used to explore a state space (such as a physical map or a logical decision tree) to find the optimal path to a designated goal.

Search State
    An abstracted representation of an environment that retains *only* the details and variables strictly necessary for an AI algorithm to plan a path or reach a goal.

Self-Attention
    The core mechanism of a Transformer, in which each token's vector is updated using the other tokens in the same sequence. Each token's query is compared (by dot product) with every token's key; softmax turns the scores into weights; and the token's new vector is the weighted sum of the values. Produces context-aware representations.

Self-Consistency
    A technique that samples several chain-of-thought responses to the same question at a nonzero temperature and selects the most common final answer by majority vote. Improves accuracy on questions with a single checkable answer at the cost of multiple calls.

Self-Supervised Learning
    A form of learning in which training labels are generated automatically from the data itself (e.g., an Autoencoder reconstructing its own input, or an LLM predicting the next word), so no human labeling is required.

Semantic Search
    Search that ranks documents by similarity of meaning rather than exact word matches, by comparing the vector embedding of the query with the embeddings of the documents. Often combined with keyword search, which handles exact strings such as codes and names better.

SHAP (SHapley Additive exPlanations)
    A mathematical method used in Explainable AI (XAI) based on cooperative game theory. It assigns a specific numerical value to each input feature, representing exactly how much that feature contributed to pushing the model's final prediction away from the baseline average.

Short-Term Memory (Agentic)
    An AI agent's immediate, temporary memory, strictly limited by the model's context window. It tracks the immediate sequence of thoughts, actions, and observations during the current operational loop.

Shortcut Learning
    A failure mode where a model learns an easy but spurious pattern that happens to correlate with the labels in the training data (e.g., snowy backgrounds instead of the animal itself), then fails when that correlation does not hold in the real world.

Sigmoid Function
    A mathematical function that squashes any real number into a valid probability value bounded strictly between 0.0 and 1.0.

Smoothing (HMM)
    The inferential task of computing a distribution over a past hidden state given evidence accumulated up to a later time step: $P(X_k \mid e_{1:t})$ where $k < t$.

Softmax
    A function that converts a vector of real-valued scores (logits) into probabilities that are positive and sum to 1, with larger scores receiving larger probabilities. The multi-class generalization of the sigmoid function.

Software Agent
    An AI program that lacks physical actuators and exists entirely digitally, calculating results or taking actions based purely on data inputs.

Space Complexity
    A mathematical Big-O representation of how much maximum memory (RAM) an algorithm requires during its execution, usually dictated by the maximum size of the frontier.

Spatial Autocorrelation
    The tendency of nearby locations to have similar values. In spatial machine learning, randomly splitting neighboring tiles between training and test sets leaks near-duplicate information and inflates test accuracy; data should instead be split by region.

Start State
    The initial configuration or state in which a rational agent begins a search problem or sequence of actions.

State Space
    The complete mathematical set of all possible states (configurations) that can exist within a given environment.

State-Value Function
    The expected return an agent will accumulate starting in state $s$ and strictly following policy $\pi$ until the episode terminates.

Stateless API
    An API in which each request is handled independently, with no memory of previous requests. LLM chat APIs are stateless, so the client must resend the full conversation history with every call.

Stochastic Environment
    An environment where the outcomes of actions are not strictly deterministic; instead, actions have probabilistic results governed by chance (e.g., weather patterns, sensor degradation, or electronic warfare jamming success rates).

Stochastic Policy
    A policy that assigns mathematical probabilities to various actions available in a state, rather than prescribing a single definitive action.

Stride
    A hyperparameter defining how many pixels a convolution filter or pooling window shifts at a time as it slides across an image or feature map.

Structured Output
    Model output in a machine-readable format such as JSON, requested so a program can use it directly. The output must still be parsed and validated, because a model may return malformed or incomplete data.

Successor Function
    The critical mapping mechanism in a search problem. It accepts a current state and a valid action, and returns the resulting next state alongside its associated path cost.

Supervised Fine-Tuning (SFT)
    Further training of a pre-trained model on curated example prompt/response pairs, using the same next-token prediction objective. The first stage of instruction tuning; it teaches the model the format and habit of answering.

Supervised Learning
    A machine learning paradigm where the algorithm is trained on historical data that already includes the correct answers (labels). It learns to map inputs to those known outputs.

Sycophancy
    The tendency of instruction-tuned LLMs to agree with or flatter the user, for example by confirming a claim embedded in a leading question. Reduced by asking neutral questions and explicitly requesting criticism.

Symbolic AI
    An early paradigm of AI that relied exclusively on explicit, hardcoded logic and symbol processing (like `if-then` rules) rather than learning from data.

System Prompt
    Standing instructions sent to an LLM, separately from the user's messages, that set its role, tone, format, and rules for the whole conversation.

Task, Format, Voice, and Context (TFVC) Framework
    A prompt-writing framework that asks the writer to state four things explicitly: the Task (what to do, using a specific verb), the Format (structure and length of the output), the Voice (role, style, and tone), and the Context (audience, purpose, constraints, and source material).

Temperature (Sampling)
    A generation setting that divides the logits before softmax. Low temperature sharpens the distribution toward the most likely token (more consistent output); high temperature flattens it (more varied output).

Testing Set
    A portion of the dataset (usually 20%) that is strictly hidden from the machine learning algorithm during training. It is used as a final "combat simulation" to test how well the model generalizes to unseen data.

Thought (ReAct)
    The step in the ReAct loop where the agent explicitly articulates its internal reasoning, assessing the current situation and determining what action to take next.

Throughput
    The number of inputs a deployed model can process per unit of time, such as frames per second. For one input at a time, throughput is roughly 1,000 divided by the inference latency in milliseconds.

Time Complexity
    A mathematical Big-O representation of how long an algorithm takes to find a solution, typically measured by the total number of nodes generated or expanded in the search tree.

Token Embedding
    The learned lookup table that maps each token ID to a dense vector; in code, a row lookup in a matrix of shape (vocabulary size, model dimension). The first layer of a Transformer.

Tokenization
    The process of splitting raw text into tokens (often sub-word chunks learned with byte-pair encoding) and mapping each token to an integer ID so it can be processed by a neural network. Tokens are the unit of an LLM's context window, speed, and API cost.

Top-k Retrieval
    Returning the k stored chunks with the highest similarity scores to the query. In RAG, k sets how much retrieved text is placed in the prompt, trading the chance of including the answer against token cost and distraction.

Top-p (Nucleus) Sampling
    A generation setting that samples the next token only from the smallest set of tokens whose probabilities add up to p, cutting off the long tail of unlikely tokens.

Training Set
    The portion of the dataset (usually 80%) exposed to the machine learning algorithm. The model uses this data to adjust its internal math and learn patterns.

Transfer Learning
    A rapid deployment technique where a massive neural network, pre-trained on millions of generic data points, has its base layers "frozen" and its final classification head re-trained for a specific tactical domain.

Transformer
    A breakthrough neural network architecture that relies heavily on self-attention and positional encoding to process sequential data (like text) entirely in parallel, forming the architectural foundation of all modern Large Language Models.

Transformer Block
    The repeated unit of a Transformer: multi-head self-attention and a feed-forward network, each wrapped with a residual connection and layer normalization. Models stack dozens of these blocks.

Transition Function
    The mathematically defined probability $P(s' \mid s,a)$ (or $T(s,a,s')$) that executing a specific action $a$ in a current state $s$ will successfully lead to the resulting state $s'$.

Transition Model (HMM)
    The conditional distribution specifying the probability of transitioning from one hidden state to another across consecutive time steps: $P(X_t \mid X_{t-1})$.

Translation Equivariance
    The property that shifting the input shifts the output by the same amount. Convolutional layers are translation-equivariant because the same filter is applied at every position: a feature is detected wherever it appears, and the feature map records where.

Translation Invariance
    The property of producing the same output when the input is shifted. In CNNs, max pooling gives tolerance only to small shifts within each pooling window; tolerance to larger shifts builds up across many layers and global pooling. CNNs are not perfectly shift-invariant. Compare {term}`Translation Equivariance`.

Underfitting (High Bias)
    A failure state where a model is too rigid or simple to learn the underlying patterns in the training data, resulting in poor predictive performance across the board.

Uninformed Search
    A class of search algorithms that explores a state space blindly, having no knowledge of where the goal is located other than the start state, valid actions, and path costs.

Uniform Cost Search (UCS)
    An uninformed search algorithm that expands the node with the lowest cumulative path cost ($g(n)$) using a Priority Queue. It is complete and guaranteed to find the optimal path.

Unsupervised Learning
    A machine learning paradigm where the algorithm is given raw, unlabeled data and must autonomously discover hidden structures, clusters, or patterns on its own.

Validation Set
    A portion of the data held out from weight updates and evaluated during training to tune settings, compare models, and trigger early stopping. Because it is used to make decisions, a separate {term}`Testing Set` must still be reserved for the final evaluation.

Vanishing Gradient
    A training failure in deep networks where gradients shrink toward zero as they are multiplied backward through many layers (especially with sigmoid activations), so the earliest layers barely learn. Mitigated by ReLU, careful initialization, normalization, and residual connections.

Variance (Statistical Error)
    The error introduced when a model is overly sensitive to tiny fluctuations or noise in the training data. High variance leads to {term}`Overfitting`.

Vector Database
    A database that stores vector embeddings together with their source text and metadata and supports fast similarity search, usually through approximate nearest neighbor indexes. The retrieval component of most RAG systems.

Vector Embeddings
    A dense array of floating-point numbers representing the core semantic meaning of a piece of text (or image), allowing algorithms to measure the "distance" or similarity between distinct concepts mathematically.

Viterbi Algorithm
    A dynamic programming algorithm for finding the most likely sequence of hidden states (the Viterbi path) that results in a sequence of observed emissions in an HMM.

Weight Sharing
    The CNN property of applying the same filter weights at every location in the input. It drastically reduces the number of parameters and lets the network recognize a feature wherever it appears.

Weights
    The learnable parameters ($w_i$) in a neural network that determine the importance or influence of a specific input feature on the final prediction.

World State
    A representation of an environment that includes every last physical detail (e.g., exact geometry, colors, irrelevant background noise). Typically too complex for efficient AI search, requiring abstraction.

Zero-Shot Prompting
    Asking a Large Language Model to perform a task without providing it with any examples of the desired output in the prompt.

Zero-Sum Game
    A mathematical representation of a situation in adversarial search (like Minimax) where one agent's gain is exactly balanced by the opponent's loss.

```