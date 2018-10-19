from abstract_curve import AbstractCurve
class Multiphase(AbstractCurve):
  "Definitions of multiphase. Returns a phase given a time."

  def __init__(self, phaseVector, nc):
    AbstractCurve.__init__(self,phaseVector[0].tmin(), phaseVector[-1].tmax())
    self.phaseVector = phaseVector
    self.nc = nc  # Dimension of a single contact constraint

  def __call__(self, t):
    assert(t>=self.tmin() and t<=self.tmax())
    for phase in self.phaseVector:
      if t>=phase.tmin() and t<=phase.tmax():
        return phase(t)

  def dim(self, t):
    assert(t>=self.tmin() and t<=self.tmax())
    for phase in self.phaseVector:
      if t>=phase.tmin() and t<=phase.tmax():
        return phase.dim(t)
