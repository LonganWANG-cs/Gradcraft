# Gradcraft Deep Learning Framework

Welcome to **Gradcraft**, a custom deep learning framework designed to provide a flexible and intuitive platform for building and training neural networks. Whether you're a beginner exploring the fundamentals of deep learning or an experienced practitioner pushing the boundaries of innovation, Gradcraft has you covered.

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to configure the environment.

```bash
pip install numpy
pip install matplotlib
pip install graphviz
```

## Features

### Core Components

- **valuecraft.py**: The engine of Gradcraft, defining data structures for gradient computation and building the backbone for reverse propagation.

- **netcraft.py**: Constructs the basic building blocks of neural networks, including single neurons, network layers, and Multi-Layer Perceptron (MLP) networks.

- **optcraft.py**: Implements optimization algorithms such as Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam) to efficiently train your models.

- **losscraft.py**: Provides fundamental loss functions like Mean Squared Error (MSE), Cross-Entropy (CE), and softmax activation.

- **piccraft.py**: Enables visualization of neural network structures through image rendering.

### Additional Utilities

- **util/hebb.py**: Implements the Hebbian learning algorithm.
  
- **util/perception_machine.py**: Defines the Perceptron learning algorithm.

- **util/performance_metrics.py**: Offers a variety of evaluation metrics, including accuracy, precision, recall, F1 score, confusion matrix, ROC AUC score, and more.

## Usage
Here's a basic example to get you started with creating and training a simple neural network using Gradcraft, and details are in the following subsections:

``` python
import random
import numpy as np
from Gradcraft.valuecraft import Value, WInitializer as WI
from Gradcraft.netcraft import MLP
from Gradcraft.piccraft import *
from sklearn.datasets import make_moons
from Gradcraft.optcraft import Optimizer

np.random.seed(1337)
random.seed(1337)

# Define the loss function
def loss(batch_size=None):
    # Set up the data loader
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # Forward computation
    scores = list(map(model, inputs))
    
    # Define the SVM "max-margin" loss function
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # Calculate accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

# Load the data
X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1  # make y be -1 or 1

# Create the model (a list represents the network architecture, followed by activation functions)
model = MLP([2, 16, 16, 1], activation='relu', init_method=WI.Kaiming_normal())  # 2-layer neural network
# model = MLP([2,16,16, 1],activation='tanh',init_method= WI.Random()) # 2-layer neural network

# Create the optimizer
# Create an SGD optimizer, pass the model's parameters to it, and set the learning rate decay parameters
sgd_optimizer = Optimizer(model.parameters(), initial_learning_rate=1, optimizer_type='sgd', decay_rate=0.9, decay_steps=100)
# Create an Adam optimizer, pass the model's parameters to it, and set the learning rate decay parameters
adam_optimizer = Optimizer(model.parameters(), initial_learning_rate=1, optimizer_type='adam', decay_rate=0.9, decay_steps=100)
# Choose to use the SGD optimizer
optimizer = sgd_optimizer
# optimizer = adam_optimizer

# Model training
for k in range(100):
    # Forward
    total_loss, acc = loss()
    # Backward
    model.zero_grad()
    total_loss.backward()
    # Update (SGD)
    optimizer.step()
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc * 100}%")

# Save the graph image
file_path = "./graph_output"
dot = draw_dot(total_loss)
save(dot, file_path)

# Final result
total_loss, acc = loss()
print(total_loss, acc)
```
## Value - Automatic Differentiation Toolkit
`Value` is a simple automatic differentiation toolkit designed to compute gradients of mathematical expressions. By building a computation graph, `Value` tracks the computation process, allowing users to easily perform reverse-mode automatic differentiation to compute gradients.
- ### Initialization
    ```python
    v = Value(data, _children=(), _op='', label='')
    ```
  - **data**: Parameter value
  - **_children**: Child nodes
  - **_op**: Operator
  - **label**: Operator label
- ### Operator Overloading
  - **__add__(self, other)**: Overload addition operator
  - **__mul__(self, other)**: Overload multiplication operator
  - **__pow__(self, other)**: Overload exponentiation operator
  - **log(self)**: Natural logarithm function
  - **exp(self)**: Exponential function
  - **tanh(self)**: Hyperbolic tangent activation function
  - **relu(self)**: ReLU activation function
  - **__neg__(self)**: Negation
  - **__radd__(self, other)**: Right addition, other + self
  - **__sub__(self, other)**: Overload subtraction operator
  - **__rsub__(self, other)**: Right subtraction, other - self
  - **__rmul__(self, other)**: Right multiplication, other * self
  - **__truediv__(self, other)**: Overload division operator
  - **__rtruediv__(self, other)**: Right division, other / self
- ### Activation Functions
  - **tanh(self)**: Hyperbolic tangent activation function
  - **relu(self)**: ReLU activation function
- ### Backward Propagation
  - **backward(self)**: Execute backward propagation to compute gradients
- ### Maximum Value Function
    ```python
    Vmax(value_list)
    ```
  - **value_list**: List of `Value` objects
  - Returns the `Value` object corresponding to the maximum value and the index of the maximum value

- ### Usage Example
    ```python
    from Value import Value, Vmax

    # Define variables
    x = Value(2)
    y = Value(3)

    # Build expression
    z = x + y
    result = z * y

    # Perform backward propagation to compute gradients
    result.backward()

    # Print results
    print(f"Result: {result.data}")
    print(f"Gradient of x: {x.grad}")
    print(f"Gradient of y: {y.grad}")
    ```

## Network - Neural Network Design
### Module
The `Module` class serves as the base class for all network modules and provides common functionality for managing gradients.
- **zero_grad(self):** Reset gradients to zero at each iteration due to accumulation.
- **parameters(self):** Return an empty list of parameters.
### Neuron
The `Neuron` class represents an individual neuron in the network.
- **__init__(self, nin, nonlin=True, activation='tanh', init_method=WI.Random()):** Initialize a neuron with the specified parameters.
- **__call__(self, x):** Output of a single neuron for a given input.
- **parameters(self):** Return the parameters of a single neuron.
- **__repr__(self):** Output format representation of the neuron.
### Layer
The `Layer` class represents a layer in the neural network, composed of multiple neurons.
- **__init__(self, nin, nout, **kwargs):** Initialize a layer with the specified number of input and output neurons.
- **__call__(self, x):** Output of all neurons in a layer for a given input.
- **parameters(self):** Return the parameters of all neurons in a layer.
- **__repr__(self):** Output format representation of the layer.
### MLP (Multi-Layer Perceptron)
The `MLP` class represents a fully connected neural network composed of multiple layers.
- **__init__(self, nLs, activation='tanh', init_method=WI.Random()):** Initialize the neural network with the specified architecture.
- **__call__(self, x):** Output of the network for a given input.
- **parameters(self):** Return all parameters of the neural network.
- **__repr__(self):** Output format representation of the neural network.
- **Usage**: A simple example is below.
  ```python
  # Create an MLP with architecture [input_size, hidden_size, output_size], activation='tanh', init_method=WI.Random()
  model = MLP([input_size, hidden_size, output_size], activation='tanh', init_method=WI.Random())

  # Forward pass
  output = model(input_data)

  # Accessing model parameters
  params = model.parameters()
  ```

## Initialization
**WInitializer**: This is a Python toolkit based on the `Value` class, designed to provide users with a variety of weight initialization methods. It includes a rich set of initialization approaches similar to `torch.nn.init.uniform`, allowing users to choose more suitable weight initialization methods based on specific use cases.

- **Consistency with PyTorch**: WInitializer maintains consistent naming and interface conventions with PyTorch's `torch.nn.init.uniform`, enabling users familiar with PyTorch to seamlessly integrate and extend their existing workflows.

- **Diversity**: Offers a variety of weight initialization methods, including uniform distribution, normal distribution, Kaiming initialization (uniform distribution), Kaiming initialization (normal distribution), and Xavier initialization. Provides users with greater convenience and flexibility.

- **Usage**: Simple examples are below.
  
    ```python
    from Gradcraft.valuecraft import WInitializer as WI
    #All of these methods are invoked through the creation of a Multi-Layer Perceptron (MLP), which represents a fully connected neural network

    #1. uniform distribution (where 'a' represents the lower bound and 'b' represents the upper bound)"
    n = MLP([3,4,4,1],activation='relu',init_method= WI.UniformInit(a=-1,b=1))

    #2.normal distribution (where 'mean' represents the mean and 'var' represents the variance)"
    n = MLP([3,4,4,1],activation='relu',init_method= WI.Normal(mean=0,var=1))

    #3.Kaiming initialization (uniform distribution)
    n = MLP([3,4,4,1],activation='relu',init_method= WI.Kaiming_uniform())

    #4.Kaiming initialization (normal distribution)
    n = MLP([3,4,4,1],activation='relu',init_method= WI.Kaiming_normal())

    ```
## Hebb
**Hebb** is a enclosed and light-weight framework, designed to construct a simple one-layer network. It has implemented model definition and training process, allowing users to quickly construct a network based on hebb learning theory.

- **Convenience**: In just **three** steps, users can construct a network based on hebb theory.
- **Error detect**: If users input more than one layer, it can raise error.
- **Diversity**: It supports supervision hebb and unsupervision hebb for users to choose.
- **Difference with netcraft**: hebb is a independent part. Because initialization strategies are for activation function, optimization strategies are for backward propagation, to avoid confusions, We set hebb up as a independent part.
- **Usage**: A simple example is below.

    ```python
    import numpy as np
    from Gradcraft.utils.hebb import Sup_Hebb


    # print function
    def graph_print(x):
        for i in range(0, 6):
            for j in range(0, 5):
                print(" ", end="")
                if x[i, j] == 1:
                    print("#", end="")
                else:
                    print(" ", end="")
            print("\n")
            
    # data
    data = np.array(
        [
            [-1,1,1,1,-1],
            [1,-1,-1,-1,1],
            [1,-1,-1,-1,1],
            [1,-1,-1,-1,1],
            [1,-1,-1,-1,1],
            [-1,1,1,1,-1],
        ]
    )
    data = data.reshape(-1, 1)
    label = data.copy()

    # usage
    hebb = Sup_Hebb([30, 30])
    result = hebb(data, label).reshape(6, 5)
    graph_print(result)
    ```
## Perception
Such as Hebb, **Perceptron** is a enclosed and light-weight framework, designed to construct a simple one-layer network. It has implemented model definition and training process, allowing users to quickly construct a network based on Perceptron learning theory.
- **Convenience**: In just **three** steps, users can construct a network based on Perceptron theory.
- **Error detect**: If users input more than one layer, it can raise error.
- **Difference with netcraft**: Perceptron is a independent part. Because initialization strategies are for activation function, optimization strategies are for backward propagation, to avoid confusions, We set Perceptron up as a independent part.
- **Usage**: A simple example is below.

  ```python
  import numpy as np
  from Gradcraft.utils.perceptron import Perceptron


  # data
  data = np.array([
      [1, 1, -1, -2],
      [1, 2, -2, -1],
  ])

  label = np.array([
      [1, 1, 0, 0]
  ])

  # usage
  perceptron = Perceptron([2, 1])
  result = perceptron(data, label)
  print(result)
  ```
## Optimization
Gradcraft provides two optimizers: Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam).
- **Initialization**
  - **__init__(self, parameters, initial_learning_rate=0.01, optimizer_type='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=0.9, decay_steps=100):** 
      - parameters: List of parameters to be optimized.
      - initial_learning_rate: Initial learning rate, default is 0.01.
      - optimizer_type: Type of optimizer, supports 'sgd' and 'adam', default is 'sgd'.
      - beta1: Exponential decay rate for the first moment estimates in Adam, default is 0.9.
      - beta2: Exponential decay rate for the second moment estimates in Adam, default is 0.999.
      - epsilon: Small value to avoid division by zero errors, default is 1e-8.
      - decay_rate: Decay rate for learning rate, default is 0.9.
      - decay_steps: Number of steps for learning rate decay, default is 100.
- **Methods**
  - **step(self):** Performs one optimization step, updating the parameters.
  - **sgd_update(self, param):** Updates parameters using Stochastic Gradient Descent.
  - **adam_update(self, param, idx):** Updates parameters using the Adam optimizer.
- **Usage Example**
  ```python
  from Gradcraft.optcraft import Optimizer

  # Create an SGD optimizer
  sgd_optimizer = Optimizer(parameters, initial_learning_rate=0.01, optimizer_type='sgd', decay_rate=0.9, decay_steps=100)

  # Create an Adam optimizer
  adam_optimizer = Optimizer(parameters, initial_learning_rate=0.01, optimizer_type='adam', decay_rate=0.9, decay_steps=100)

  # Choose to use the SGD optimizer
  optimizer = sgd_optimizer
  # Alternatively, choose to use the Adam optimizer
  # optimizer = adam_optimizer

  # Execute an optimization step in the training loop
  for epoch in range(num_epochs):
      # Other training steps
      # ...
      # Perform one optimization step
      optimizer.step()
  ```
## Performance Metrics
**Performance_metrics** provides various evaluation metrics to assess model performance for both Classification and Regression tasks, and offers functions for visualizing results. It includes a range of functions similar to sklearn.metrics, allowing users to select appropriate metrics conveniently.
### Classification metrics   
- **pr_ap_score(y_true, y_pred)**: this function computes the average precision (AP) from prediction scores. The value is between 0 and 1 and higher is better. 
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)   
  - **Returns**:   
    - **ap**: the average precision (float)  
  - **Usage**: A simple example is below.
    ```python
    from Gradcraft.utils.performance_metrics import pr_ap_score
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    ap = pr_ap_score(ys, ypred)
    print("Ap:", ap)    # Ap: 0.41666666666666663
    ```   
- **accuracy_score(y_true, y_pred)**: this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. 
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)    
  - **Usage**: A simple example is below.
    ```python
    from Gradcraft.utils.performance_metrics accuracy_score
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    accuracy = accuracy_score(y_true, y_pred)    
    ```  

- **precision_score(y_true, y_pred)**: the precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives 
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)   
  - **Returns**:   
    - **precision**: precision of the positive class in binary classification or weighted average of the precision of each class for the multiclass task.  
  - **Usage**: A simple example is below.
    ```python
    from Gradcraft.utils.performance_metrics import precision_score
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    precision = precision_score(ys, ypred)
    ```
- **recall_score(y_true, y_pred)**: The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)   
  - **Returns**:   
    - **recall** : recall of the positive class in binary classification or weighted average of the recall of each class for the multiclass task.
  - **Usage**: A simple example is below.
    ```python
    from Gradcraft.utils.performance_metrics import recall_score
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    recall = recall_score(ys, ypred)
    ```
- **f1_score(y_true, y_pred)**: The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. 
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)   
  - **Returns**:   
    - **f1**: F1 score of the positive class in binary classification or weighted average of the F1 scores of each class for the multiclass task.  
  - **Usage**: A simple example is below.  
    ```python
    from Gradcraft.utils.performance_metrics import f1_score
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    f1 = f1_score(ys, ypred)
    ```

- **confusion_matrix(y_true, y_pred)**: this function computes confusion matrix to evaluate the accuracy of a classification.r. 
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)   
  - **Returns**:   
    - **cm** : confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
  - **Usage**: A simple example is below.  
    ```python
    from Gradcraft.utils.performance_metrics import confusion_matrix
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    cm = confusion_matrix(ys, ypred)
    ```

- **roc_auc_score(y_true, y_pred)** : this function computes Compute confusion matrix to evaluate the accuracy of a classification.
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)   
  - **Returns**:   
    - **auc**: the area under the curve. (float)  
  - **Usage**: A simple example is below.  
    ```python
    from Gradcraft.utils.performance_metrics import roc_auc_score
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    auc = roc_auc_score(ys, ypred)
    ```

### Regression metrics 
- **mean_squared_error(cm, classes)** : Mean squared error regression loss.
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)     
  - **Returns**:   
    - **mse**: A non-negative floating point value (the best value is 0.0), or an array of floating point values, one for each individual target.
  - **Usage**: A simple example is below.  
    ```python
    from Gradcraft.utils.performance_metrics import mean_squared_error
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    mse = mean_squared_error(ys, ypred)
    ```   
- **mean_absolute_error(cm, classes)** : Mean absolute error regression loss.
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)    
  - **Returns**:   
    - **mae**: output is non-negative floating point. The best value is 0.0.
  - **Usage**: A simple example is below.  
    ```python
    from Gradcraft.utils.performance_metrics import mean_absolute_error
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    mae = mean_absolute_error(ys, ypred)

    ```   
- **r2_score(cm, classes)**:  R2(coefficient of determination) regression score function.Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).  
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
    - **y_pred**: predicted labels (list)     
  - **Returns**:    	
    - **r2**: The r2 score or ndarray of scores if `multioutput` is `raw_values`.
  - **Usage**: A simple example is below. 
    ```python
    from Gradcraft.utils.performance_metrics import r2_score 
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    r2 = r2_score(ys, ypred)
    ```   

### Plotting
- **plot_confusion_matrix(cm, classes)** this function can be used to visually represent a confusion matrix.  
  - **Parameters**:   
    - **cm**: confusion matrix      
    - **classes**: predicted labels (list)   
  - **Usage**: A simple example is below. 
    ```python
    from Gradcraft.utils.performance_metrics import plot_confusion_matrix 
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    cm = confusion_matrix(ys, ypred)
    plot_confusion_matrix(cm, [-1, 1])
    ```   

- **plot_roc_curve(y_true, y_pred, pos_label=1, show_auc=True)** : this function calculates and visualize ROC curve (currently only support binary classification case).
  - **Parameters**:   
    - **y_true**: ground truth labels (list)      
  - **y_pred**: predicted labels (list)   
    - **pos_label (default: 1)**: the class considered as the positive class when computing the roc auc metrics   
    - **show_auc (default: True)**: area under ROC curve
  - **Usage**: A simple example is below. 
    ```python
    from Gradcraft.utils.performance_metrics import plot_roc_curve
    ys = [-1, 1, 1, -1]
    ypred = [1, 1, 1, 1]
    plot_roc_curve(ys, ypred)
    ```   

## Loss
**LossCraft** is a Python toolkit based on the Value class, designed to create personalized loss functions. It includes a variety of loss functions similar to `torch.nn.functional`, allowing users to easily construct custom loss functions tailored to their specific tasks.

- **Customizability**: Build flexible and personalized loss functions for your tasks using the Value class.

- **Consistency with PyTorch**: LossCraft maintains consistent naming and interface conventions with PyTorch's torch.nn.functional, enabling users familiar with PyTorch to seamlessly integrate and extend their existing workflows.

- **Supported Loss Functions**: LossCraft includes commonly used loss functions such as Mean Squared Error (MSELoss) and Cross-Entropy Loss (CrossEntropyLoss).

- **Utility Functions**: The toolkit also provides practical functions like softmax, log_softmax, and nll_loss, facilitating the construction of more complex loss functions.

- **Usage**: A simple example is below.
  
```python
from Gradcraft.valuecraft import Value, Vmax
from Gradcraft.losscraft import MSELoss, CrossEntropyLoss

# MSELoss
mse_loss = MSELoss(reduction='mean')
pred = [Value(1), Value(2), Value(3), Value(4)]
target = [Value(4), Value(3), Value(2), Value(1)]
loss = mse_loss(pred, target)

# CrossEntropyLoss
cross_entropy_loss = CrossEntropyLoss(reduction='mean')
pred = [[Value(0.2), Value(0.8)], [Value(0.7), Value(0.3)]]
target = [[Value(0), Value(1)], [Value(1), Value(0)]]
loss = cross_entropy_loss(pred, target)
```

## Visualization
Gradcraft provides a simple and effective graph visualization tool using Graphviz. This tool allows users to visualize the computation graph of their neural network, including nodes representing operations, values, and their connections.
- **Graph Visualization Functions**
  - **trace(root)**
    - **Description:** Traces the computation graph starting from the root node.
    - **Input:** `root` - The root node of the computation graph.
    - **Output:** Returns a set of nodes and a set of edges representing the computation graph.
  - **draw_dot(root, format='svg', rankdir='LR')**

    - **Description:** Creates a Graphviz Digraph object representing the computation graph.
    - **Input:**
      - `root` - The root node of the computation graph.
      - `format` - Output format for the visualization (default is 'svg').
      - `rankdir` - Direction of graph layout ('LR' for left to right, 'TB' for top to bottom).
    - **Output:** Returns a Graphviz Digraph object.

  - **save(dot, file_path, file_format='svg')**

    - **Description:** Saves the generated Graphviz Digraph to a file.
    - **Input:**
      - `dot` - The Graphviz Digraph object.
      - `file_path` - Path to save the visualization file.
      - `file_format` - File format for the saved visualization (default is 'svg').

- **Usage**: A simple example is below.
  ```python
  from Gradcraft.piccraft import draw_dot, save

  # Assuming 'root' is the root node of the computation graph
  dot = draw_dot(root, format='svg', rankdir='LR')

  # Save the visualization to a file
  save(dot, file_path='./graph_output', file_format='svg')
  ```
## Contributors
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
Please make sure to update tests as appropriate.












