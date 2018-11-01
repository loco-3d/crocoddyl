from abstract_curve import AbstractCurve
class Phase(AbstractCurve):
  "Definitions of a phase. Returns contact information between two time intervals"

  def __init__(self, indices, tmin, tmax):
    AbstractCurve.__init__(self,tmin, tmax)

    self.indices = indices
    self._dim = len(indices)
    
  def __call__(self, t):
    assert(t>=self.tmin() and t<=self.tmax())
    return self.indices

  def dim(self, t):
    return self._dim
