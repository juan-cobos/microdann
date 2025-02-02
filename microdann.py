import random
import math
from engine import Value
from nn import Module, Layer
import time

class Dendrite(Module):
    
    def __init__(self, nin): 
        self.sw = [Value(random.uniform(-1, 1)) for _ in range(nin)] # SW : Synaptic Weights
        self.b = Value(0)

    def __call__(self, x, rf=16):
        """
        List with index of RFs for that dendrite. First with random sampling TODO: Local Receptive Fields

        """
        random_mask = random.sample(range(len(x)), rf) if len(x) >= rf else range(len(x)) 
        self.m = [1 if i in random_mask else 0 for i in range(len(x))] # Boolean mask of RFs 
        self.filt_sw = [swi * mi for swi, mi in zip(self.sw, self.m)]
        out = sum([swi*xi for swi, xi in zip(self.filt_sw, x)], self.b)
        return out.relu()
    
    def parameters(self):
        return self.sw + [self.b] # TODO: Used params are only RFs, not all weights 

    def __repr__(self):
        return f"Dendrite({len(self.sw)})"

class Soma(Module): 

    def __init__(self, nin, n_dends):
        self.cw = [Value(random.uniform(-1, 1)) for _ in range(n_dends)]
        self.b = Value(0)
        self.dends = [Dendrite(nin) for _ in range(n_dends)] 
    
    def __call__(self, x):
        # TODO: Add boolean mask to somatas
        out = sum([d(x) * cwi for d, cwi in zip(self.dends, self.cw)], self.b)
        return out.relu()

    def parameters(self):
        return [p for d in self.dends for p in d.parameters()] + self.cw + [self.b] 

    def __repr__(self):
        return f"Soma({len(self.cw)})"

class DANN(Module):

    def __init__(self, nin, n_dends, n_somas, nout):
        """
        
        Args:
            nin (int): number of inputs
            n_dends (int): number of dendrites per neuron
            n_somas (int): number of independent somas
            nout (_type_): number of neurons in the output layer
        """
        self.somas = [Soma(nin, n_dends) for s in range(n_somas)]
        self.out_layer = Layer(n_somas, nout)

    def __call__(self, x):
        logits = self.out_layer([s(x) for s in self.somas])
        probs = [xj.exp()/sum([xi.exp() for xi in logits]) for xj in logits]
        return probs

    def parameters(self):
        return [p for s in self.somas for p in s.parameters()] + self.out_layer.parameters()

    def __repr__(self):
        return f"DANN([{len(self.cw)})"