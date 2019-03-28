import numpy as np
from collections import OrderedDict

class EESplines(OrderedDict):
  def __add__(self,other):
    return EESplines([[patch, self[patch] + other[patch]] for patch in self.keys()]);
  def __sub__(self,other):
    return EESplines([[patch, self[patch] - other[patch]] for patch in self.keys()]);

class CentroidalPhi:

  class zero():
    def __init__(self, dim):
      self.dim = dim
    def eval(self, t): return (np.matrix(np.zeros((self.dim,1))),
                               np.matrix(np.zeros((self.dim,1))))

  def __init__(self, com_vcom=None, hg=None, forces=None):
    if com_vcom is None: self.com_vcom = self.zero(6)
    else:                self.com_vcom = com_vcom
    if hg is None:       self.hg = self.zero(6)
    else:                self.hg = hg
    if forces is None:   self.forces = EESplines()
    else:                self.forces = forces
    
  def __add__(self,other):
    if isinstance(self.com_vcom,self.zero):
      return CentroidalPhi(other.com_vcom, other.hg, other.forces)
    if isinstance(other.com_vcom,other.zero):
      return CentroidalPhi(self.com_vcom, self.hg, self.forces)
    return CentroidalPhi(self.com_vcom+other.com_vcom,self.hg+other.hg,
                         self.forces+other.forces)
  def __sub__(self,other):
    if isinstance(self.com_vcom,self.zero):
      return NotImplementedError
    if isinstance(other.com_vcom,other.zero):    
      return NotImplementedError
    return CentroidalPhi(self.com_vcom-other.com_vcom,self.hg-other.hg,
                      self.forces-other.forces)
