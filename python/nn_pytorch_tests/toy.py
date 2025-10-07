from siren import SIRENLayer
from nir import NIRLayer, NIRTrunk, MultiHeadNIR


class Tester:
    def __init__(self, start, activation, params):
        super().__init__()
        self.start = start
        self.activation = activation
        self.params = params
        self.activation(self, ("start",), *self.params)

def linear(self, vars, a, b):
    result = a * getattr(self, vars[0]) + b
    setattr(self, vars[0], result)

tester = Tester(3.14, linear, (2.0, 1.0))
print(tester.start)


w0 = 30.0
layer_counts = (256,)*5
siren = MultiHeadNIR(SIRENLayer, in_dim=3, layer_counts=layer_counts, params=(w0,))
