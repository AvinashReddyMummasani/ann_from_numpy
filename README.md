# Simple Neural Network from Scratch

Hey! This is a project where I built a multi-layer neural network using only **NumPy**. No massive frameworks like PyTorch or TensorFlow—just math, matrices, and a bit of Python. 

I made this to really understand how things like backpropagation and weight updates work under the hood.

## What's inside?

- **Custom Layers**: You can define as many layers and neurons as you want.
- **Activation Functions**: It supports `ReLU`, `Sigmoid`, and `Linear`.
- **Loss Options**: Choose between `MSE` (for regression) or `BCE` (for classification).
- **Training**: It uses Stochastic Gradient Descent to learn from your data.

## How to use it

First, make sure you have the dependencies:
`pip install numpy tqdm`

### 1. Set up the network
You just need to define your dimensions, the activations for each layer, and the loss type. 

```python
from your_script_name import Neural_network

# Example: 2 inputs, one hidden layer with 4 neurons, and 1 output
nn = Neural_network(
    dim=[2, 4, 1], 
    activations=['relu', 'sigmoid'], 
    loss='mse', 
    learning_rate=0.01
)
