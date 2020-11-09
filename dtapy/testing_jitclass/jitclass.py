import numpy as np
import numba as nb
from collections import OrderedDict

my_array=np.arange(50000).reshape(10000,5)

capacity_column=4

spec_network=OrderedDict
spec_network={'links':Links}
@nb.jitclass(spec_network)
class Network(spec_network):
    def  __init__(self, link_matrix):
        self.links=Links(link_matrix)



spec_link=OrderedDict
spec_link={'capacity':np.ndarray}
@nb.jitclass(spec_link)
class Links(spec_link):
    def __init__(self, link_matrix):
        self.capacity=link_matrix[:,capacity_column]
        


