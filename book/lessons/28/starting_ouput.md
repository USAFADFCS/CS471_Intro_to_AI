
```markdown
# 2.3 Neural Networks I: Architectures and Forward Propagation

In the previous lessons, we used statistical models to draw lines and curves through data. But what happens when the operational environment becomes so complex that no single mathematical curve can separate a threat from a friendly asset?

To solve highly non-linear problems, we look to biological inspiration. The human brain does not calculate a single formula; it processes signals through billions of interconnected neurons. In this lesson, we introduce the **Artificial Neural Network (ANN)**, the foundation of modern Deep Learning, and explore how stacking simple mathematical operations creates systems capable of mastering multi-domain warfare.

## The Artificial Neuron (The Perceptron)

The building block of a neural network is the artificial neuron, often called a perceptron. It performs three distinct steps:
1.  **Receive Inputs:** It takes in numerical features ($x_1, x_2, \dots, x_n$).
2.  **Compute Weighted Sum:** It multiplies each input by a specific weight ($w_1, w_2, \dots, w_n$), sums them up, and adds a bias term ($b$).
3.  **Apply Activation:** It passes the sum through an "activation function" to decide if the neuron should "fire."

The mathematical formula for the weighted sum $z$ is:
$$z = (w_1 x_1 + w_2 x_2 + \dots + w_n x_n) + b$$

Or in compact linear algebra notation (dot product):
$$z = W \cdot X + b$$

---

## Simple Problem: Single Perceptron

**Scenario:** UAV Sortie Viability. You need to predict if a small reconnaissance UAV has enough battery to complete its sortie based on three features: Payload Weight, Headwind Speed, and Distance. 

### The Math: Step Activation Function
For a basic perceptron, the activation function is a simple threshold. If the weighted sum is greater than 0, the neuron fires (Predicts 1: Success). If it is less than or equal to 0, it stays silent (Predicts 0: Failure).

$$f(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

> **Tip: Toy Problem - The Perceptron**
> You are evaluating a UAV sortie. 
> Inputs: Payload ($x_1 = 5\text{ kg}$), Headwind ($x_2 = 10\text{ knots}$), Distance ($x_3 = 20\text{ km}$).
> Trained Weights: $w_1 = -2$ (weight drains battery), $w_2 = -1$ (wind drains battery), $w_3 = -0.5$ (distance drains battery).
> Bias: $b = 40$ (Base battery capacity score).
> 
> 1. Calculate weighted sum: $z = (-2 \times 5) + (-1 \times 10) + (-0.5 \times 20) + 40$
> 2. Simplify: $z = -10 - 10 - 10 + 40 = 10$
> 3. Apply Step Activation: Since $10 > 0$, the output is **1**.
> 
> **Result:** The perceptron fires. The UAV has enough battery to complete the mission.

### Python Lab: UAV Sortie Prediction

In this lab, we will use Python to simulate a single perceptron making decisions on UAV telemetry.

* **Generate Data:** Create a synthetic dataset of UAV flights (features: payload, wind, distance) and outcomes (Success/Failure).
* **Build the Perceptron:** Use `scikit-learn`'s Perceptron model.
* **Train and Predict:** Fit the model to find the optimal weights.
* **Visualize:** Extract the learned weights to see exactly how the network values each feature.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate UAV Telemetry Data
np.random.seed(42)
n_samples = 500
payload = np.random.uniform(2, 10, n_samples)
headwind = np.random.uniform(0, 25, n_samples)
distance = np.random.uniform(10, 50, n_samples)

# Create a rule for Success (1) or Failure (0) with some noise
# Formula: Base capacity - (payload*2) - (headwind*1) - (distance*0.5) > 0
base_capacity = 45
z_true = base_capacity - (payload * 2) - (headwind * 1) - (distance * 0.5) + np.random.normal(0, 3, n_samples)
y = (z_true > 0).astype(int)

X = np.column_stack((payload, headwind, distance))
df_uav = pd.DataFrame(X, columns=['Payload_kg', 'Headwind_kts', 'Distance_km'])
df_uav['Sortie_Success'] = y

print("Sample UAV Telemetry Data:")
print(df_uav.head(), "\n")

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Perceptron
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train, y_train)

# 4. Evaluate and Visualize Weights
y_pred = perceptron.predict(X_test)
print(f"Perceptron Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")

weights = perceptron.coef_[0]
bias = perceptron.intercept_[0]

features = ['Payload', 'Headwind', 'Distance']
plt.figure(figsize=(8, 4))
plt.bar(features, weights, color=['darkred', 'red', 'lightcoral'])
plt.axhline(0, color='black', linewidth=1)
plt.title(f'Learned Network Weights (Bias = {bias:.2f})')
plt.ylabel('Weight Value (Negative = Drains Battery)')
plt.show()

```

### Interpreting the Results

A single perceptron is effectively a linear classifier. The bar chart exposes the network's "brain." Because the generated data penalized payload the most heavily, the perceptron mathematically learned to assign the largest negative weight to the payload feature. If the sum of these negative impacts outweighs the positive bias (the base battery), the output drops below zero, and the model predicts failure.

---

## Moderate Problem: Forward Propagation & Non-Linearity

**Scenario:** Sonar Classification. You are tracking a submarine. A single linear perceptron cannot separate the complex acoustic returns of a metallic submarine hull from the clutter of underwater rocks. We must stack neurons into a Multi-Layer Perceptron (MLP).

### The Math: Hidden Layers and Activation Functions

When we stack neurons, the outputs of the first layer become the inputs to the next. The layer between the input and output is called a **Hidden Layer**.

Crucially, we must apply **non-linear activation functions** in the hidden layers. If we only use linear math ($y = mx+b$), stacking layers does nothing—math proves that a linear function of a linear function is just one big linear function. Non-linearity allows the network to fold and curve its decision boundaries.

**1. ReLU (Rectified Linear Unit):** Used in hidden layers. It outputs the input directly if positive; otherwise, it outputs zero.


$$\text{ReLU}(z) = \max(0, z)$$

**2. Sigmoid:** Used in the final output layer for binary classification to squash the final number into a probability percentage between 0 and 1.


$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

> **Tip: Toy Problem - Forward Propagation**
> Let's trace a signal through a tiny network.
> Input feature: $x = 2$.
> **Hidden Neuron 1:** Weight $w_1 = -1.5$, Bias $b_1 = 1$.
> * Calculate $z_1 = (-1.5 \times 2) + 1 = -3 + 1 = -2$.
> * Apply ReLU: $\max(0, -2) = 0$. The neuron is dead.
> 
> 
> **Hidden Neuron 2:** Weight $w_2 = 3$, Bias $b_2 = -2$.
> * Calculate $z_2 = (3 \times 2) - 2 = 6 - 2 = 4$.
> * Apply ReLU: $\max(0, 4) = 4$. The neuron fires a signal of 4.
> 
> 
> **Output Neuron:** Takes inputs from Hidden 1 and 2. Weights: $w_{o1} = 2, w_{o2} = 0.5$. Bias $b_o = -1$.
> * Calculate $z_o = (2 \times 0) + (0.5 \times 4) - 1 = 0 + 2 - 1 = 1$.
> * Apply Sigmoid: $\sigma(1) = \frac{1}{1 + e^{-1}} \approx 0.73$.
> 
> 
> **Result:** The network predicts a 73% chance of a Target.

### Python Lab: Tracing the Signal

In this lab, we will build a 2-layer neural network from absolute scratch using matrix multiplication in NumPy to classify sonar returns.

* **Generate Data:** Simulate the "Mines vs. Rocks" sonar dataset (non-linear data).
* **Define Math Functions:** Write the ReLU and Sigmoid functions in raw Python.
* **Initialize Weights:** Create weight matrices for the hidden layer and the output layer.
* **Execute Forward Propagation:** Push the data through the network step-by-step and calculate the final predictions.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. Generate Non-Linear Sonar Data (Interlocking crescent shapes)
X, y = make_moons(n_samples=400, noise=0.15, random_state=42)

plt.figure(figsize=(6, 4))
plt.scatter(X[y==0, 0], X[y==0, 1], label='Rock (Clutter)', color='blue', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], label='Submarine (Target)', color='red', alpha=0.6)
plt.title("Sonar Acoustic Returns")
plt.legend()
plt.show()

# 2. Define Activation Functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 3. Initialize Network Architecture (2 Inputs -> 5 Hidden Neurons -> 1 Output)
np.random.seed(42)
W1 = np.random.randn(2, 5) # Weights from Input to Hidden
b1 = np.zeros((1, 5))      # Biases for Hidden
W2 = np.random.randn(5, 1) # Weights from Hidden to Output
b2 = np.zeros((1, 1))      # Bias for Output

# 4. Execute Forward Propagation
# Step A: Input to Hidden Layer
Z1 = np.dot(X, W1) + b1
A1 = relu(Z1) # The activation introduces non-linearity!

# Step B: Hidden Layer to Output
Z2 = np.dot(A1, W2) + b2
A2 = sigmoid(Z2) # Final probability

# Convert probabilities to binary predictions
predictions = (A2 > 0.5).astype(int).flatten()

# Note: Because the weights are random (we haven't trained it yet), 
# the accuracy will be poor. Training is covered in Lesson 2.4.
print(f"Untrained Network Accuracy: {np.mean(predictions == y):.2f}")
print("First 5 Output Probabilities:")
print(A2[:5].flatten())

```

### Interpreting the Results

The scatter plot shows interlocking data. No single straight line can separate the rocks from the submarines.

By executing `np.dot(X, W1)`, the network creates 5 different linear boundaries simultaneously. By passing those through the `relu` function, we bend and cut those lines. By combining those bent lines in the final output layer `np.dot(A1, W2)`, the network is capable of drawing a complex, custom-fit polygon around the target data. While the accuracy above is low because the weights are randomized (training and backpropagation happen in the next lesson), the mathematical architecture required to solve the problem is fully operational.

---

## Complex Problem: Multi-Output Architectures

**Scenario:** Flight Dynamics. A commander wants to know two things simultaneously: How much fuel will the strike package consume, and exactly how many minutes until they are over the target?

### The Math: Vector Outputs

Statistical models typically output a single number. Neural Networks can output an entire vector simply by adding more neurons to the final layer. If the final layer has two neurons with linear activations (no Sigmoid, because we are predicting raw continuous numbers, not probabilities), the network produces two distinct predictions from the exact same shared hidden layers.

$$Y = [y_{\text{fuel}}, y_{\text{time}}]$$

### Python Lab: Multi-Target Regression

In this lab, we will use `scikit-learn`'s Multi-Layer Perceptron to predict two separate operational metrics simultaneously.

* **Generate Data:** Simulate altitude, throttle, and drag data.
* **Target Variables:** Calculate fuel consumed and time-on-target.
* **Build Architecture:** Deploy a neural network with 2 hidden layers (64 neurons each) and an output layer of 2.
* **Evaluate:** Plot the predictions for both metrics.

```python
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 1. Generate Flight Dynamics Data
n_flights = 1000
altitude = np.random.uniform(10000, 40000, n_flights)
throttle = np.random.uniform(50, 100, n_flights)
drag_coeff = np.random.uniform(0.02, 0.05, n_flights)

X_flight = np.column_stack((altitude, throttle, drag_coeff))

# Complex formulas for the targets
fuel_consumed = (throttle ** 1.5) * (drag_coeff * 100) - (altitude * 0.001) + np.random.normal(0, 5, n_flights)
time_to_target = 6000 / (throttle * (1 - drag_coeff)) + (altitude * 0.005) + np.random.normal(0, 2, n_flights)

Y_flight = np.column_stack((fuel_consumed, time_to_target))

# Split
X_tr, X_te, Y_tr, Y_te = train_test_split(X_flight, Y_flight, test_size=0.2, random_state=42)

# 2. Build and Train Multi-Output Neural Network
# 2 hidden layers of 64 neurons each. Output layer automatically sizes to Y (2).
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=500, random_state=42)
mlp.fit(X_tr, Y_tr)

# 3. Predict and Evaluate
Y_pred = mlp.predict(X_te)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Fuel Prediction
axes[0].scatter(Y_te[:, 0], Y_pred[:, 0], alpha=0.5, color='orange')
axes[0].plot([Y_te[:,0].min(), Y_te[:,0].max()], [Y_te[:,0].min(), Y_te[:,0].max()], 'k--')
axes[0].set_title('Fuel Consumed (Predicted vs Actual)')
axes[0].set_xlabel('Actual Fuel')
axes[0].set_ylabel('Predicted Fuel')

# Plot Time Prediction
axes[1].scatter(Y_te[:, 1], Y_pred[:, 1], alpha=0.5, color='teal')
axes[1].plot([Y_te[:,1].min(), Y_te[:,1].max()], [Y_te[:,1].min(), Y_te[:,1].max()], 'k--')
axes[1].set_title('Time to Target (Predicted vs Actual)')
axes[1].set_xlabel('Actual Time')
axes[1].set_ylabel('Predicted Time')

plt.tight_layout()
plt.show()

```

### Interpreting the Results

Look at the plots. The dotted black line represents perfect predictions. The tight clustering of the orange and teal dots along the line proves that the network successfully learned to predict two highly distinct physical phenomena simultaneously.

By sharing the hidden layers, the network learns an internal representation of "aerodynamics." It figures out how throttle and drag interact in the hidden layers, and then branches that shared knowledge out to the two specific output neurons. This is highly efficient for complex multi-domain modeling.

---

## Operational Readiness: Practice Problems

**1. Weighted Sum Calculation**
An artificial neuron receives three sensor inputs: $x_1 = 4$, $x_2 = 2$, $x_3 = -1$. The corresponding trained weights are $w_1 = 0.5$, $w_2 = -2.0$, $w_3 = 3.0$. The bias is $b = 1.5$. Calculate the final weighted sum ($z$).

* **Step 1:** Multiply inputs by weights. $(4 \times 0.5) = 2$, $(2 \times -2.0) = -4$, $(-1 \times 3.0) = -3$.
* **Step 2:** Sum the results. $2 - 4 - 3 = -5$.
* **Step 3:** Add the bias. $-5 + 1.5 = -3.5$.
* **Answer:** The weighted sum $z$ is **-3.5**.

**2. ReLU Activation Step**
Take the weighted sum $z = -3.5$ calculated in Problem 1. If this neuron is in a hidden layer using a ReLU activation function, what is the final output of the neuron?

* **Step 1:** Apply the ReLU function formula: $\max(0, z)$.
* **Step 2:** Evaluate $\max(0, -3.5)$. Since -3.5 is less than 0, the maximum is 0.
* **Answer:** The output is **0** (the neuron is dead/inactive).

**3. Sigmoid Activation Calculation**
An output neuron calculates a weighted sum of $z = 2.0$. It uses a Sigmoid activation function to classify a radar contact as a threat. Calculate the probability percentage output by the neuron.

* **Step 1:** Apply the Sigmoid formula: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
* **Step 2:** Substitute $z = 2.0$. $\frac{1}{1 + e^{-2.0}}$.
* **Step 3:** Calculate (using $e^{-2} \approx 0.135$). $\frac{1}{1 + 0.135} = \frac{1}{1.135} \approx 0.881$.
* **Answer:** The network outputs an **88.1%** probability.

**4. Forward Propagation (Hidden Node)**
You are tracing forward propagation for a simple network. Input vector $X = [3, -2]$. The weights connecting the input to Hidden Node A are $W_A = [1.0, 0.5]$. The bias is $b_A = -1.0$. Calculate the output of Hidden Node A assuming a ReLU activation function.

* **Step 1:** Calculate dot product (weighted sum). $z_A = (3 \times 1.0) + (-2 \times 0.5) - 1.0$.
* **Step 2:** Simplify. $z_A = 3.0 - 1.0 - 1.0 = 1.0$.
* **Step 3:** Apply ReLU. $\max(0, 1.0) = 1.0$.
* **Answer:** The output of Hidden Node A is **1.0**.

**5. Forward Propagation (Output Node)**
Continuing from Problem 4, Hidden Node A outputs $1.0$, and a separate Hidden Node B outputs $3.0$. These signals are sent to the final Output Node. The weights to the output node are $W_o = [-2.0, 1.5]$. The bias is $b_o = 0.5$. Calculate the raw weighted sum ($z$) of the output node.

* **Step 1:** Calculate the weighted sum of the incoming hidden signals. $z_o = (1.0 \times -2.0) + (3.0 \times 1.5) + 0.5$.
* **Step 2:** Simplify. $z_o = -2.0 + 4.5 + 0.5$.
* **Step 3:** Final sum. $z_o = 3.0$.
* **Answer:** The raw weighted sum is **3.0**.

**6. Counting Architectural Parameters**
You are designing a feedforward neural network to analyze 5 different weather telemetry inputs. The network has one hidden layer with 10 neurons, and an output layer with 2 neurons. Calculate the total number of learnable parameters (weights + biases) in this network.

* **Step 1:** Calculate Input-to-Hidden parameters. 5 inputs $\times$ 10 neurons = 50 weights. Each hidden neuron has 1 bias = 10 biases. Total = 60.
* **Step 2:** Calculate Hidden-to-Output parameters. 10 hidden signals $\times$ 2 output neurons = 20 weights. Each output neuron has 1 bias = 2 biases. Total = 22.
* **Step 3:** Sum the parameters. $60 + 22 = 82$.
* **Answer:** The network has **82** trainable parameters.

```

```