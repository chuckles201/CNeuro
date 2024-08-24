from datatype import CValue
import random
from neural_viz import draw_trace, viz_network
import pdb

class Neuron: # takes in values and computes weighted sum with activation func
    def __init__(self, nin):
        self.w = [CValue(random.uniform(-1,1)) for _ in range(nin)]
        self.b = CValue(random.uniform(-1,1))
        self.params = self.w + [self.b] # getting out all parameters (to GD. later!)
        self.data = 0.0
        
    def __call__(self,x):
        out = sum([xi*wi for xi,wi in zip(x,self.w)],self.b)
        self.data = out.tanh()
        return self.data

class Layer: # makes a layer of n neurons, outputing n values and taking in o values
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] # neuron for each nout
        self.params = [p for n in self.neurons for p in n.params]
        
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out if len(out) != 1 else out[0]

class MLP: # taking each layer and giving its output to another layer (neural net!)
    def __init__(self,nin,lshape):
        self.dims = [nin] + lshape
        self.layers = [Layer(self.dims[i],self.dims[i+1]) for i in range(len(lshape))] # full list, creating layers!
        self.params = [p for l in self.layers for p in l.params]
        self.input = None
    def __call__(self,X):
        out = X
        self.input = X
        for layer in self.layers:
            out = layer(out)
        return out

def gradient_descent(model,X,y,epochs): # simple GD algorithm
    for epoch in range(epochs):
        ypreds = []
        for i in range(len(X)):
            ypreds.append(model(X[i]))
        mse = sum([(yp - yi) ** 2 for yp,yi in zip(ypreds,y)])
        mse.backward()
        
        for p in model.params:
            p.data -= p.grad *0.01
            p.grad = 0 # zeroing out gradient!
        print(f'epoch {epoch} mse: {mse}')
    viz_network(model,y[len(y)-1])
    draw_trace(mse)
    return mse

# example simple problem!
m = MLP(2,[10,4,3,5,4,1])
y_real = [-1,-0.72,0.8,0.75]
xs = [[2,4],[3,5],[-10,1],[3,0]]
g = gradient_descent(m,xs,y_real,1000)
