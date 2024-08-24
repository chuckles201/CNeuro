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
            dot.node(vid,label = "%s | %.4f"%(n.data,n.grad),shape='polygon')
            dot.node(oid,label = "%s"%(n._op))
            dot.edge(oid,vid)
        else:
            vid = str(id(n))
            dot.node(vid,label = "%s | %.4f"%(n.data,n.grad),shape='polygon')
    for e in edges: # creating edges between child and intermed parent
        id1 = str(id(e[0]))
        id2 = str(id(e[1])) + e[1]._op
        dot.edge(id1,id2)
    return dot

a = CValue(4)
y = CValue(5,label='y')
b = a-y
z = 3 * b; z.label = 'z'
c = z**2
c.backward()
d = draw_trace(c)

with open('graph.dot', 'w') as f: # writing to file
    f.write(str(d))
