'''
Creating a 'value' class. 
The reason it is necessary to create this class, is beacuase in neural networks,
we don't just only compute the values and loss function. We also need to remember all the
operations that the values went through (from the first weights and inputs on), so we
can compute the local derivatives of each parameter relative to the final derivative
of the loss (with the chain rule).

This will also allow us to vizualize all the operations that are preformed, and that result
in 1 final value.

We should be able to vizualize the entire process of the neural network creation, the code should be clean, and it should also
exactly match pytorch (w/ functionality such as backward...)
'''
import math

class CValue:
    def __init__(self,data,children=set(),label='',op='',backward=None):
        self.data =data # numerical value
        self._prev = children # values that created value
        self._backward = backward # derivative of childs' of value
        self.label = label # label assigned to value
        self._op = op # operation that created value
        self.grad = 0.0
        
    def __repr__(self):
        return f'CValue:({self.data})'
    
    def __add__(self,other): # add, backward function distributes gradient
        other = other if isinstance(other,CValue) else CValue(other)
        out = CValue(self.data + other.data,children=(self,other),op='+') # creating new value with children
        
        def backward(): # backward for addition maintains gradient
            other.grad += out.grad * 1
            self.grad += out.grad * 1
        out._backward = backward
        
        return out
    
    def __sub__(self,other):
        return self + -other
    
    def __neg__(self): # -self
        return self * -1
    
    def __radd__(self,other): # other + self
        return self + other
    
    def __mul__(self,other): # same as addition, just different gradient and operation
        other = other if isinstance(other,CValue) else CValue(other)
        out = CValue(self.data * other.data,children=(self,other),op='*') # creating new value with children
        
        def backward(): # backward for addition maintains gradient
            other.grad += out.grad * self.data
            self.grad += out.grad * other.data
        out._backward = backward
        
        return out
    
    def __rmul__(self,other):
        return self * other
    
    def __pow__(self,other):#making sure that the data is not a value class
        assert isinstance(other,(int,float)), "Powers must be intergers or floats for now..."
        
        out = CValue(self.data ** other,children=(self,),op=f'pow{other}') # creating new value with 1 child
        
        def backward(): # backward for power 
            self.grad += out.grad * ((self.data ** (other-1)) * other)
        out._backward = backward
        
        return out
    
    def tanh(self):
        out = CValue(math.tanh(self.data))
        def backward():
            self.grad += out.grad * (math.tanh(self.data)**2 -1)
            
        out._backward = backward
        return out
        
        
    def __truediv__(self,other):
        return self * (other ** -1)

    def backward(self): # create topological tree, and then backpropogate through!
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1 # auto gradient of answer is 1
        for val in reversed(topo): # backward func reversed through tree (all values with least parents, then children!)
            if val._backward is not None: # if not a leaf node
                val._backward()
            else: 
                pass
            
    