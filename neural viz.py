'''
Vizualizing the neural network by getting the topological values, 
and connecting all of the children to all of the parents. This should
make it so that we can vizualize what is happening to the 'tree' of operations'
'''
from datatype import CValue
import graphviz

def trace(root): # if value not already iterated, adding edges to children, adding node, and building children
    nodes = set()
    edges = set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes,edges # topological nodes, all edges


def draw_trace(root):
    nodes,edges = trace(root)
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
    for n in nodes:
        if len(n._prev) != 0: # creating operator node and (out) value node
            vid = str(id(n))
            oid = str(id(n)) + n._op
            dot.node(vid,label = "{%s | %.4f}"%(n.data,n.grad),shape='polygon')
            dot.node(oid,label = "%s"%(n._op))
            dot.edge(oid,vid)
        else:
            vid = str(id(n))
            dot.node(vid,label = "{%s | %.4f}" % (n.data,n.grad),shape='polygon')
    for e in edges: # creating edges between child and intermed parent
        id1 = str(id(e[0]))
        id2 = str(id(e[1])) + e[1]._op
        dot.edge(id1,id2)
    with open('graph.dot', 'w') as f: # writing to file
        f.write(str(dot))
    return 'done'





'''
Now, visualizing an actual neural network by just the neurons and their connections/outputs
'''

def viz_network(mlp,expected):
    dot = graphviz.Digraph(graph_attr={'rankdir': 'LR'})
    for i in range(len(mlp.layers) + 1): # for all layers adding nodes and edges of previous
        if i == 0:
            for x in mlp.input:
                idi = str(id(x))
                dot.node(idi,label=f"x->{x}",shape='polygon')
        elif i ==1:
            i1 = 0
            for neuron in mlp.layers[i-1].neurons: # for each layer add previous, or input
                idn = str(id(neuron))
                dot.node(idn,f'nrn{i},{i1}',shape='circle')
                i1 += 1
                for x in mlp.input:
                    dot.edge(str(id(x)),idn) # previous id with current id
        else:
            i1 = 0
            i2 = 0
            for neuron in mlp.layers[i-1].neurons:
                idn = str(id(neuron))
                dot.node(idn,f'nrn{i},{i1}',shape='circle')
                i1 += 1
                for e in mlp.layers[i-2].neurons: # adding to all prev neurons
                    dot.edge(str(id(e)),idn)
                    i2 += 1
            if (len(mlp.layers) - i) == 0:
                for neuron in mlp.layers[i-1].neurons:
                    idd = str(id(neuron.data))
                    dot.node(idd,label=f'output:{neuron.data}',shape='polygon')
                    dot.edge(str(id(neuron)),idd)
                dot.node('expected',f"Expected: {expected}",shape='plaintext')
    with open('neuralviz.dot', 'w') as f: # writing to file
        f.write(str(dot))
  
    
                
            
