import numpy as np

# https://youtu.be/w8yWXqWQYmU?si=MXhI9EgsfYXMdshP&t=917

#function type
# 1 = relu
# 2 = sigmoid
# 3 = tanh
# 4 = softmax
func_type = 1

#Import Data
with open(r"C:\Users\John\Desktop\Albany\Machine Learning\Project\data.txt", newline='') as csvfile:
    data = list(csv.reader(csvfile))
    
#break data into training and development
m, n = data.shape

dev = data[0:1000]
Y_dev = dev[0]
X_dev = dev[1:n]

train = data[1000:m]
Y_dev = train[0]
X_dev = train[1:n]

# initialize w1,b1,w2,b2
# w1 shape (input_size, hidden_size)
# b1 shape (1, hidden_size)
# w2 shape (hidden_size, output_size)
# b2 shape (1, output_size)
def std_initialize_parameters(input_size, hidden_size, output_size):
    w1 = randn(hidden_size, input_size)
    b1 = zeros(hidden_size, 1)
    w2 = randn(output_size, hidden_size)
    b2 = zeros(output_size, 1)
    return w1, b1, w2, b2

def he_initialize_parameters(input_size, hidden_size, output_size):
    w1 = randn(hidden_size, input_size) * sqrt(2/ input_size)
    b1 = zeros(hidden_size, 1)
    w2 = randn(output_size, hidden_size) * sqrt(2/ hidden_size)
    b2 = zeros(output_size, 1)
    return w1, b1, w2, b2

# forward propagation
# calculate z1, a1, z2, a2
# return
def forward_propagation(x, w1, b1, w2, b2):
    z1 = w1*x + b1
    a1 = activation_function(z1)
    z2 = w2*x + b2
    a2 = activation_function(z2)
    return z1, a1, z2, a2

# activation function
# relu, sigmoid, tanh, softmax
def activation_function(z, func_type)
    if func_type == 1
        return activation_function_relu(z)
    if func_type == 2
        return activation_function_sigmoid(z)    
    if func_type == 3
        return activation_function_tanh(z)    
    if func_type == 4
        return activation_function_softmax(z)

def activation_function_sigmoid(z):
    return 1/(1-exp(z))

def activation_function_relu(z):
    return maximum(0,z)

def activation_function_tanh(z):
    return (exp(2x)-1)/(exp(2x)+1)

def activation_function_softmax(z):
    return exp(z)/sum(exp(z))

# derivation of activation function
# relu, sigmoid, tanh, softmax
def activation_function_derivative(z, func_type)
    if func_type == 1
        return activation_function_derivative_relu(z)
    if func_type == 2
        return activation_function_derivative_sigmoid(z)    
    if func_type == 3
        return activation_function_derivative_tanh(z)    
    if func_type == 4
        return activation_function_derivative_softmax(z)
    
def activation_function_derivative_sigmoid(z):
    return exp(z)*(sigmoid(z)**2)

def activation_function_derivative_relu(z):
    return z > 0

def activation_function_derivative_tanh(z):
    return 4/(exp(z)+exp(-z))**2
    
def activation_function_derivative_softmax(z):
    return softmax(z)*(1-softmax(z))

# one-hot encoding
def one_hot(y):
    one_hot_y = zeroes(m)
    one_hot_y[y] = 1
    return one_hot_y

# backward propagation
# calculate dw1, db1, dw2, db2
def backward_propagation(x, y, z1, a1, z2, a2, w2):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1/m * dz2.dot(a1)
    db2 = 1/m * sum(dz2)
    dz1 = w2.dot(dz2)*activation_fucntion_derivative(z1)
    dw1 = 1/m * dz1.dot(x)
    db1 = 1/m * dum(dz1)
    return dw1, db1, dw2, db2

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    return w1, b1, w2, b2

def get_predictions(a2):
    return max(a2, 0)


def get_accuracy(y_pred, y_true):
    return sum(y_pred == y_true) / y_true.size

# in each iteration
# forward propagation
# backward propagation
# update parameters
# print the accuracy and loss
def gradient_descent(x, y, w1, b1, w2, b2, learning_rate, num_iterations):
     initialize_parameters
    for i in range(iterations)
        z1, a1, z2, a2 = forward_propagation(x, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward_propagation(x, y, z1, a1, z2, a2)
        w1, b1, w2, b2 = update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate)
    return w1, b1, w2, b2

def make_prediction(X, w1, b1, w2, b2):
    return

def test_model(X, y, w1, b1, w2, b2):
    return
