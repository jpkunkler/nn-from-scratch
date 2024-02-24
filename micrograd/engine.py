import math

class Value:

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0 # at initialization, we assume that every value does not effect the output
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value=({self.data})"
    
    def tanh(self):
        '''
        Hyperbolic Tangent (tanh) Activation Function
        '''
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad # Chain rule with the derivative of tanh, where t is our tanh value
        out._backward = _backward
        
        return out
    
    def relu(self):
        '''
        Rectified Linear Units (ReLU) Activation Function
        '''
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        '''
        Sigmoid Activation Function (Logistic Function)
        '''
        x = -1.0 * self.data
        s = 1 / (1 + math.exp(x)) # 1 / (1 + e**-x)
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += (s * (1 - s)) * out.grad
        out._backward = _backward

        return out
        
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self): # initialize backward function to be called on the final output node to iterate backwards through all children
        self.grad = 1.0
        topo = []
        visited = set()
        def buildTopo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    buildTopo(child)
                topo.append(v)
        
        buildTopo(self)
        
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # if we receive a non-value attribute, convert it to value
        out = Value(self.data + other.data, (self, other), "+") # for every addition, feed in the children to the new value as well as the operation type

        def _backward():
            self.grad += 1.0 * out.grad # in an addition, out.grad is simply copied to the input/child nodes
            other.grad += 1.0 * out.grad # this is according to the chain rule
        out._backward = _backward # dont call the function!! this would return None. Only store the function reference, therefore no ()

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (float, int)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f"**{other}")

        def _backward():
            self.grad += other * (self.data**(other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # if we receive a non-value attribute, convert it to value
        out = Value(self.data * other.data, (self, other), "*") # for every multiplication, feed in the children to the new value as well as the operation type

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __radd__(self, other): # other + self
        return self + other
    
    def __truediv__(self, other): # self / other
        return self * other**-1 # NOTE: For this to work, we need to implement the power function for our Value object
    
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    # --- IMPLEMENT SUBTRACTION BY MULTIPLICATION WITH -1; WHY? To use the stuff we already built. :) ---
    def __neg__(self):
        return self * -1 # since we implemented the __mul__ function, we can do this

    def __sub__(self, other):
        return self + (-other) # tada, implemented subtraction with a smart hack :)
    
    def __rsub__(self, other):
        return self + (-other)
    # -------------------------------------------------------