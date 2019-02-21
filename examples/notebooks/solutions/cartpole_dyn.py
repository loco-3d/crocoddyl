# Use this function inside DifferentialActionModel.calc by setting:
#      data.xout[:]  = carpole_dynamics(model,data,x,u)
def cartpole_dynamics(model,data,x,u):
    # Getting the state and control variables
    x,th,xdot,thdot = x
    f, = u

    # Shortname for system parameters
    m1,m2,l,g = model.m1,model.m2,model.l,model.g
    s,c = np.sin(th),np.cos(th)

    # Defining the equation of motions
    m = m1+m2
    mu = m1+m2*s**2
    xddot  = (f     + m2*c*s*g - m2*l*s*thdot**2 )/mu
    thddot = (c*f/l + m*g*s/l  - m2*c*s*thdot**2 )/mu

    return [ xddot,thddot ]

