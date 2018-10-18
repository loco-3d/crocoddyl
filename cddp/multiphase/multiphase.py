from abstract_curve import AbstractCurve
class Multiphase(AbstractCurve):
  "Definitions of multiphase. Returns a phase given a time."

  def __init__(self, phaseVector):
    AbstractCurve.__init__(self,phaseVector[0].tmin(), phaseVector[-1].tmax())
    self.phaseVector = phaseVector

  def __call__(self, t):
    assert(t>=self.tmin() and t<=tmax())
    for phase in phaseVector:
      if t>=phase.tmin() and t<=phase.tmax():
        return phase(t)
