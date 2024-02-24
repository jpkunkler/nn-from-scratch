import random
from micrograd.engine import Value

class Neuron:
    '''
    One single Neuron in a Neural Network.
    Each Neuron accepts nin incoming values.
    For each incoming value, one weight is created.
    Each Neuron has a single bias value.

    The Neuron returns the value of its activation function by multiplying each input value times the corresponding weight, calculating the sum and adding the bias.
    '''
    def __init__(self, nin): # nin = number of inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)] # initialize weights; we need as many weights as the number of inputs
        self.b = Value(random.uniform(-1,1)) # initialize the bias

    def __call__(self, x):
        # w * x + b aka the forward pass
        # act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b # long version
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # smart version; more efficient since sum does not start at 0.0 but at bias
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    '''
    A Layer is a collection of Nodes at a specific stage/depth in a Neural Network.
    The Number of Inputs (nin) determines the number of incoming connections, i.e. the quantity of Nodes in a previous layer. If the previous layer has 3 Neurons, each Neuron within this new Layer will receive three input values.
    The Number of Outputs (nouts) determines the number of Neurons inside the layer, i.e. how many values will be passed along by the layer.
    '''
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs # if only one value in output, i.e. final output layer, return the value
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    '''
    Simple Implementation of a Multi-Layer Perceptron (MLP) based.
    '''
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # create a total list, i.e. for 3 input nodes and 2 layers of 4 nodes plus 1 output node = [3] + [4, 4, 1] = [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] # create all layers, i.e. 3 to 4, 4 to 4 and 4 to 1

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) 
            # this implements the forward pass! Start by inserting the initial input values into the first layer, receive their results, pass these results on to the next layer, receive their results and so on...
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]