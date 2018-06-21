from abc import ABCMeta, abstractmethod


class IntegratorBase:
  __metaclass__ = ABCMeta

  def __init__(self):
    pass

  abstractmethod
  def integrate(self,model,data,t,x0,u,dt):
    pass

class EulerIntegrator(IntegratorBase):

  @staticmethod
  def integrate(model,data,t,x0,u,dt):
    return x0 + model.f(x0,u,data)*dt

class RK4Integrator(IntegratorBase):

  @staticmethod
  def integrate(model,data,t,x0,u,dt):
    k1 = model.f(x0,u,data)

    x2 = x0 + dt/2.*k1
    k2 = model.f(x2,u,data)

    x3 = x0 + dt/2.*k2
    k3 = model.f(x3,u,data)

    x4 = x0 + dt*k3
    k4 = model.f(x4,u,data)
    return x0 + dt/6.*(k1 + 2.*k2 + 2.*k3 + k4)

def computeFlow(integrator,model,data,timeline,x0,controls):
  N = len(timeline)
  x_flow = [None]*N
  x_flow[0] = x0
  for k in range(len(timeline)-1):
    dt = timeline[k+1] - timeline[k]
    t = timeline[k]
    control = controls[k]
    x_flow[k+1] = integrator.integrate(model,data,t,x_flow[k],control,dt)

  return x_flow