# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 01:26:29 2019

@author: adam
"""

import random
import math
import pandas as pd
import numpy as np

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
'''
class NeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error
'''
class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs
'''
    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs
'''
class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.squash(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # Apply the logistic function to squash the output of the neuron
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    #
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

###

# Blog post example:

'''
nn = NeuralNetwork(2, 3, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3,0.4,0.5], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55,0.5,0.5], output_layer_bias=0.6)
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))
'''
# XOR example:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))


def init_weights_from_inputs_to_hidden_layer1_neurons(hidden_layer1_weights):
    weight_num = 0
    for h in range(len(hidden_layer1.neurons)):
        for i in range(num_inputs):
            if not hidden_layer1_weights:
                hidden_layer1.neurons[h].weights.append(random.random())
            else:
                hidden_layer1.neurons[h].weights.append(hidden_layer1_weights[weight_num])
            weight_num += 1
    
    
def init_weights_from_hidden_layer1_neurons_to_hidden_layer2_neurons(hidden_layer2_weights):
    weight_num = 0
    for h in range(len(hidden_layer2.neurons)):
        for i in range(num_hidden1):
            if not hidden_layer2_weights:
                hidden_layer2.neurons[h].weights.append(random.random())
            else:
                hidden_layer2.neurons[h].weights.append(hidden_layer2_weights[weight_num])
            weight_num += 1
    
    
def init_weights_from_hidden_layer2_neurons_to_output_layer_neurons(output_layer_weights):
    weight_num = 0
    for h in range(len(output_layer.neurons)):
        for i in range(num_hidden2):
            if not output_layer_weights:
                output_layer.neurons[h].weights.append(random.random())
            else:
                output_layer.neurons[h].weights.append(output_layer_weights[weight_num])
            weight_num += 1
    
def print_nerons_weights(layer):
    if layer == hidden_layer1:
        print("hidden_layer1 weughts")
    elif layer == hidden_layer2:
        print("hidden_layer2 weughts")
    else:
        print("output_layer weughts")
        
    for h in range(len(layer.neurons)):
        print(layer.neurons[h].weights)

def print_weights():
    print_nerons_weights(hidden_layer1)
    print_nerons_weights(hidden_layer2)
    print_nerons_weights(output_layer)
    


def feed_forward_algorithm(inputs):
    ##print(inputs)
    hidden_layer1_outputs = hidden_layer1.feed_forward(inputs)
    ##print(hidden_layer1_outputs)
    hidden_layer2_outputs = hidden_layer2.feed_forward(hidden_layer1_outputs)
    ##print(hidden_layer2_outputs)
    output_layer_outputs = output_layer.feed_forward(hidden_layer2_outputs)
    ##print(output_layer_outputs)
    return output_layer_outputs

def training(training_inputs , training_outputs):
    
    total_error = 0
    outputs  = feed_forward_algorithm(training_inputs)
    ##print(outputs)
    for o in range(len(outputs)):
        total_error += 0.5 * ((outputs[o] - training_outputs[o]) ** 2) 
    ##print(total_error)
    
    ##1.output neuron delta
    pd_errors_wrt_output_neuron_total_net_input = [0] * len(output_layer.neurons) ##delta o1,o2
    for o in range(len(output_layer.neurons)):
        # ∂E/∂zⱼ
        pd_errors_wrt_output_neuron_total_net_input[o] = output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

    # 2. Hidden neuron2 deltas
    pd_errors_wrt_hidden2_neuron_total_net_input = [0] * len(hidden_layer2.neurons) ##delta h21,h22,h23
    for h2 in range(len(hidden_layer2.neurons)):##

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
        d_error_wrt_hidden2_neuron_output = 0
        for o in range(len(output_layer.neurons)): ## sigma delat*weight
            d_error_wrt_hidden2_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * output_layer.neurons[o].weights[h2]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
        pd_errors_wrt_hidden2_neuron_total_net_input[h2] = d_error_wrt_hidden2_neuron_output * hidden_layer2.neurons[h2].calculate_pd_total_net_input_wrt_input()
        
    #3. Hidden neuron1 deltas
    pd_errors_wrt_hidden1_neuron_total_net_input = [0] * len(hidden_layer1.neurons)
    for h1 in range(len(hidden_layer1.neurons)):##
        d_error_wrt_hidden1_neuron_output = 0
        for h2 in range(len(hidden_layer2.neurons)):##
            d_error_wrt_hidden1_neuron_output +=  pd_errors_wrt_hidden2_neuron_total_net_input[h2] * hidden_layer2.neurons[h2].weights[h1]

        pd_errors_wrt_hidden1_neuron_total_net_input[h1] = d_error_wrt_hidden1_neuron_output * hidden_layer1.neurons[h1].calculate_pd_total_net_input_wrt_input()

    # 4. Update output neuron weights
    for o in range(len(output_layer.neurons)):
        for w_h2o in range(len(output_layer.neurons[o].weights)):

            # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
            pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_h2o)

            # Δw = α * ∂Eⱼ/∂wᵢ
            output_layer.neurons[o].weights[w_h2o] -= LEARNING_RATE * pd_error_wrt_weight
            
    # 5. Update hidden2 neuron weights
    for h2 in range(len(hidden_layer2.neurons)):
        for w_h1h2 in range(len(hidden_layer2.neurons[h2].weights)):

            # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
            pd_error_wrt_weight = pd_errors_wrt_hidden2_neuron_total_net_input[h2] * hidden_layer2.neurons[h2].calculate_pd_total_net_input_wrt_weight(w_h1h2)

            # Δw = α * ∂Eⱼ/∂wᵢ
            hidden_layer2.neurons[h2].weights[w_h1h2] -= LEARNING_RATE * pd_error_wrt_weight

    # 6. Update hidden1 neuron weights

    for h1 in range(len(hidden_layer1.neurons)):
        for w_ih1 in range(len(hidden_layer1.neurons[h1].weights)):
            
            # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
            pd_error_wrt_weight = pd_errors_wrt_hidden1_neuron_total_net_input[h1] * hidden_layer1.neurons[h1].calculate_pd_total_net_input_wrt_weight(w_ih1)
            
            # Δw = α * ∂Eⱼ/∂wᵢ
            hidden_layer1.neurons[h1].weights[w_ih1] -= LEARNING_RATE * pd_error_wrt_weight     



LEARNING_RATE = 0.1
num_hidden1 = 3
num_hidden2 =3
num_outputs = 2
num_inputs = 6
hidden_layer1_bias = 0.35
hidden_layer2_bias = 0.35
output_layer_bias = 0.35
training_inputs = [0.05,0.1,0.15,0.2,0.3,0.35]
training_outputs = [0.01 , 0.99]
result = []


##np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2. / layers_dims[l - 1])
hidden_layer1_weights = [random.uniform(0.1,0.2) * np.sqrt(2. / 27) *  np.sqrt(2. / 927) for i in range(18)]
hidden_layer2_weights = [random.uniform(0.1,0.2) * np.sqrt(2. / 9) for i in range(9)]
output_layer_weights = [random.uniform(0.1,0.2) for i in range(6)]

hidden_layer1 = NeuronLayer(num_hidden1, hidden_layer1_bias)
hidden_layer2 = NeuronLayer(num_hidden2, hidden_layer2_bias)    
output_layer = NeuronLayer(num_outputs, output_layer_bias)

init_weights_from_inputs_to_hidden_layer1_neurons(hidden_layer1_weights)
init_weights_from_hidden_layer1_neurons_to_hidden_layer2_neurons(hidden_layer2_weights)
init_weights_from_hidden_layer2_neurons_to_output_layer_neurons(output_layer_weights)

##print_nerons_weights(hidden_layer1)
##print_nerons_weights(hidden_layer2)
##print_nerons_weights(output_layer)

training_data = pd.read_csv('titanic.csv')
x_train = training_data[0:800].drop(['Survived'] ,axis=1)
x_test = training_data[800:891].drop(['Survived'] ,axis=1)
y_train = training_data.loc[0:799,['Survived']]
y_test = training_data.loc[800:890,['Survived']]
##print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)
##print(x_train.loc[0].values)


print_weights()


for index in range(len(x_train)):
    if y_train.loc[index].values == 1:
        training_outputs = [0.99,0.01]
    else:
        training_outputs = [0.01,0.99]

    for i in range(10):
        training(x_train.loc[index].values,training_outputs)
print("\n")

print_weights()

print("------------------------test---------------------------------")
for index in range(800,800+len(x_test)):
    output = feed_forward_algorithm(x_test.loc[index].values)
    print(output)
    if output[0] > output[1]:
        result.append(1)
    else:
        result.append(0)
    
print(result)












