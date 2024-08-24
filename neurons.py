from datatype import CValue
import random

class Neuron: # takes in values and computes weighted sum with activation func
    def __init__(self, nin):
        self.w = [CValue(random.uniform(-1,1)) for _ in range(nin)]
        self.b = CValue(random.uniform(-1,1))
        self.params = self.w + [self.b] # getting out all parameters (to GD. later!)
        
    def __call__(self,x):
        out = sum([xi*wi for xi,wi in zip(x,self.w)],self.b)
        return out.tanh()

class Layer: # makes a layer of n neurons, outputing n values and taking in o values
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)] # neuron for each nout
        self.params = [p for n in self.neurons for p in n.params]
        
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out

class MLP: # taking each layer and giving its output to another layer (neural net!)
    def __init__(self,nin,lshape):
        self.dims = [nin] + lshape
        self.layers = [Layer(self.dims[i],self.dims[i+1]) for i in range(len(lshape))] # full list, creating layers!
        self.params = [p for l in self.layers for p in l.params]
    def __call__(self,X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

def gradient_descent(model,X,y,epochs):
    for epoch in range(epochs):
        ypreds = []
        for i in range(len(X)):
            ypreds.append(model(X[i]))
        mse = sum([(yp - yi) ** 2 for yp,yi in zip(ypreds[0],y)])
        mse.backward()
        print(model.layers[0].neurons[0].w[0].grad)
        for p in model.params:
            p.data -= p.grad * 0.01
            p.grad = 0 # zeroing out gradient!
        print(f'epoch {epoch} mse: {mse}')


m = MLP(3,[4,4,2])
y_real = [1,-1,0]
xs = [[3,4,5],[-3,-2,-1],[1,-2,-1]]

gradient_descent(m,xs,y_real,100)


    