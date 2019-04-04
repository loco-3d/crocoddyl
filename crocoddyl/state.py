import numpy as np
import pinocchio
from scipy.linalg import block_diag

from .utils import a2m


class StateAbstract:
    r""" Abstract class for the state representation.

    A state is represented by its operators: difference, integrates and their
    derivatives. The difference operator returns the value of x1 [-] x2
    operation. Instead the integrate operator returns the value of x [+] dx.
    These operators are used to compared two points on the state manifold M or
    to advance the state given a tangential velocity (\in Tx M). Therefore
    the points x, x1 and x2 belongs to the manifold M; and dx or x1 [-] x2 lie
    on its tangential space.
    """

    def __init__(self, nx, ndx):
        # Setting up the dimension of the state vector and its tangent vector
        self.nx = nx
        self.ndx = ndx

    def zero(self):
        """ Return a zero reference state.
        """
        raise NotImplementedError("Not implemented yet.")

    def rand(self):
        """ Return a random state.
        """
        raise NotImplementedError("Not implemented yet.")

    def diff(self, x1, x2):
        r""" Operator that differentiates the two state points.

        It returns the value of x1 [-] x2 operation. Note tha x1 and x2 are
        points in the state manifold (\in M). Instead the operator result lies
        in the tangent-space of M.
        :param x1: current state
        :param x2: next state
        :return x2 [-] x1 value
        """
        raise NotImplementedError("Not implemented yet.")

    def integrate(self, x, dx):
        r""" Operator that integrates the current state.

        It returns the value of x [+] dx operation. Note tha x and dx are
        points in the state manifold (\in M) and its tangent, respectively.
        Note that the operator result lies on M too.
        :param x: current state
        :param dx: displacement of the state
        :return x [+] dx value
        """
        raise NotImplementedError("Not implemented yet.")

    def Jdiff(self, x1, x2, firstsecond='both'):
        r""" Compute the partial derivatives of difference operator.

        For a given state, the difference operator (x1 [-] x2) is defined by
        diff(x1,x2). Instead here it is described its partial derivatives, i.e.
        \partial{diff(x1,x2)}{x1} and \partial{diff(x1,x2)}{x2}. By default,
        this function returns the derivatives of the first and second argument
        (i.e. firstsecond='both'). However we ask for a specific partial
        derivative by setting firstsecond='first' or firstsecond='second'.
        :param x1: current state
        :param x2: next state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the diff(x1,x2) function
        """
        raise NotImplementedError("Not implemented yet.")

    def Jintegrate(self, x, dx, firstsecond='both'):
        r""" Compute the partial derivatives of integrate operator.

        For a given state, the integrate operator (x [+] dx) is defined by
        integrate(x,dx). Instead here it is described its partial derivatives,
        i.e. \partial{integrate(x1,x2)}{x1} and \partial{integrate(x1,x2)}{x2}.
        By default, this function returns the derivatives of the first and
        second argument (i.e. firstsecond='both'). However we ask for a specific
        partial derivative by setting firstsecond='first' or firstsecond='second'.
        :param x: current state
        :param dx: displacement of the state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the integrate(x,dx) function
        """
        raise NotImplementedError("Not implemented yet.")


class StateVector(StateAbstract):
    """ Euclidean state vector.

    For this kind of states, the difference and integrate operators are
    described by substraction and addition operations. Due to the Euclide space
    point and its velocity lie in the same space, all Jacobians are described
    throught the identity matrix.
    """

    def __init__(self, nx):
        # Euclidean point and its velocity lie in the same space dimension.
        StateAbstract.__init__(self, nx, nx)

    def zero(self):
        """ Return a null state vector.
        """
        return np.zeros([self.nx])

    def rand(self):
        """ Return a random state vector.
        """
        return np.random.rand(self.nx)

    def diff(self, x1, x2):
        """ Difference between x2 and x1.

        :param x1: current state
        :param x2: next state
        :return x2 - x1 value
        """
        return x2 - x1

    def integrate(self, x1, dx):
        """ Integration of x + dx.

        Note that there is no timestep here. Set dx = v dt if you're integrating
        a velocity v during an interval dt.
        :param x: current state
        :param dx: displacement of the state
        :return x [+] dx value
        """
        return x1 + dx

    def Jdiff(self, x1, x2, firstsecond='both'):
        """ Jacobians for the x2 - x1.

        :param x1: current state
        :param x2: next state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the diff(x1,x2) function
        """
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jdiff(x1, x2, 'first'), self.Jdiff(x1, x2, 'second')]

        J = np.zeros([self.ndx, self.ndx])
        if firstsecond == 'first':
            J[:, :] = -np.eye(self.ndx)
        elif firstsecond == 'second':
            J[:, :] = np.eye(self.ndx)
        return J

    def Jintegrate(self, x, dx, firstsecond='both'):
        """ Jacobians of x + dx.

        :param x: current state
        :param dx: displacement of the state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the integrate(x,dx) function
        """
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jintegrate(x, dx, 'first'), self.Jintegrate(x, dx, 'second')]
        return np.eye(self.ndx)


class StateNumDiff(StateAbstract):
    """ State class with NumDiff for Jacobian computation.

    This class allows us to avoid the definition of analytical Jacobians, which
    is error prone. Additionally it can be used to debug the Jacobian
    computation of a custom State. For doing so, we need to construct this class
    by passing as an argument our State object.
    """

    def __init__(self, State):
        StateAbstract.__init__(self, State.nx, State.ndx)
        self.State = State
        self.disturbance = 1e-6

    def zero(self):
        """ Return a zero reference state defined in State.
        """
        return self.State.zero()

    def rand(self):
        """ Return a random reference state defined in State.
        """
        return self.State.rand()

    def diff(self, x1, x2):
        """ Run the differentiate operator defined in State.

        :param x1: current state
        :param x2: next state
        :return x2 [-] x1 value
        """
        return self.State.diff(x1, x2)

    def integrate(self, x, dx):
        """ Run the integrate operator defined in State.

        :param x: current state
        :param dx: displacement of the state
        :return x [+] dx value
        """
        return self.State.integrate(x, dx)

    def Jdiff(self, x1, x2, firstsecond='both'):
        """ Compute the partial derivatives for diff operator using NumDiff.

        :param x1: current state
        :param x2: next state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the diff(x1,x2) function
        """
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jdiff(x1, x2, 'first'), self.Jdiff(x1, x2, 'second')]
        dx = np.zeros(self.ndx)
        h = self.disturbance
        J = np.zeros([self.ndx, self.ndx])
        d0 = self.diff(x1, x2)
        if firstsecond == 'first':
            for k in range(self.ndx):
                dx[k] = h
                J[:, k] = self.diff(self.integrate(x1, dx), x2) - d0
                dx[k] = 0
        elif firstsecond == 'second':
            for k in range(self.ndx):
                dx[k] = h
                J[:, k] = self.diff(x1, self.integrate(x2, dx)) - d0
                dx[k] = 0
        J /= h
        return J

    def Jintegrate(self, x, dx, firstsecond='both'):
        """ Compute the partial derivatives for integrate operator using NumDiff.

        :param x: current state
        :param dx: displacement of the state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the integrate(x,dx) function
        """
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jintegrate(x, dx, 'first'), self.Jintegrate(x, dx, 'second')]
        Dx = np.zeros(self.ndx)
        h = self.disturbance
        J = np.zeros([self.ndx, self.ndx])
        d0 = self.integrate(x, dx)
        if firstsecond == 'first':
            for k in range(self.ndx):
                Dx[k] = h
                J[:, k] = self.diff(d0, self.integrate(self.integrate(x, Dx), dx))
                Dx[k] = 0
        elif firstsecond == 'second':
            for k in range(self.ndx):
                Dx[k] = h
                J[:, k] = self.diff(d0, self.integrate(x, dx + Dx))
                Dx[k] = 0
        J /= h
        return J


class StatePinocchio(StateAbstract):
    """ State definition based on Pinocchio model.

    Pinocchio defines operators for integrating or differentiating the robot's
    configuration space. And here we assume that the state is defined by the
    robot's configuration and its joint velocities (x=[q,v]). Generally speaking,
    q lies on the manifold configuration manifold (M) and v in its tangent space
    (Tx M). Additionally the Pinocchio allows us to compute analytically the
    Jacobians for the differentiate and integrate operators. Note that this code
    can be reused in any robot that is described through its Pinocchio model.
    """

    def __init__(self, pinocchioModel):
        StateAbstract.__init__(self, pinocchioModel.nq + pinocchioModel.nv, 2 * pinocchioModel.nv)
        self.model = pinocchioModel

    def zero(self):
        """ Return the neutral robot configuration with zero velocity.
        """
        q = pinocchio.neutral(self.model)
        v = np.zeros(self.model.nv)
        return np.concatenate([q.flat, v])

    def rand(self):
        """ Return a random state.
        """
        q = pinocchio.randomConfiguration(self.model)
        v = np.random.rand(self.model.nv) * 2 - 1
        return np.concatenate([q.flat, v])

    def diff(self, x0, x1):
        """ Difference between the robot's states x2 and x1.

        :param x1: current state
        :param x2: next state
        :return x2 [-] x1 value
        """
        nq, nv, nx = self.model.nq, self.model.nv, self.nx
        assert (x0.shape == (nx, ) and x1.shape == (nx, ))
        q0 = x0[:nq]
        q1 = x1[:nq]
        v0 = x0[-nv:]
        v1 = x1[-nv:]
        dq = pinocchio.difference(self.model, a2m(q0), a2m(q1))
        return np.concatenate([dq.flat, v1 - v0])

    def integrate(self, x, dx):
        """ Integrate the current robot's state.

        :param x: current state
        :param dx: displacement of the state
        :return x [+] dx value
        """
        nq, nv, nx, ndx = self.model.nq, self.model.nv, self.nx, self.ndx
        assert (x.shape == (nx, ) and dx.shape == (ndx, ))
        q = x[:nq]
        v = x[-nv:]
        dq = dx[:nv]
        dv = dx[-nv:]
        qn = pinocchio.integrate(self.model, a2m(q), a2m(dq))
        return np.concatenate([qn.flat, v + dv])

    def Jdiff(self, x1, x2, firstsecond='both'):
        """ Compute the partial derivatives for diff operator using Pinocchio.

        :param x1: current state
        :param x2: next state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the diff(x1,x2) function
        """
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jdiff(x1, x2, 'first'), self.Jdiff(x1, x2, 'second')]
        if firstsecond == 'second':
            dx = self.diff(x1, x2)
            q = a2m(x1[:self.model.nq])
            dq = a2m(dx[:self.model.nv])
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return block_diag(np.asarray(np.linalg.inv(Jdq)), np.eye(self.model.nv))
        else:
            dx = self.diff(x2, x1)
            q = a2m(x2[:self.model.nq])
            dq = a2m(dx[:self.model.nv])
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return -block_diag(np.asarray(np.linalg.inv(Jdq)), np.eye(self.model.nv))

    def Jintegrate(self, x, dx, firstsecond='both'):
        """ Compute the partial derivatives for integrate operator using Pinocchio.

        :param x: current state
        :param dx: displacement of the state
        :param firstsecond: desired partial derivative
        :return the partial derivative(s) of the integrate(x,dx) function
        """
        assert (firstsecond in ['first', 'second', 'both'])
        assert (x.shape == (self.nx, ) and dx.shape == (self.ndx, ))
        if firstsecond == 'both':
            return [self.Jintegrate(x, dx, 'first'), self.Jintegrate(x, dx, 'second')]
        q = a2m(x[:self.model.nq])
        dq = a2m(dx[:self.model.nv])
        Jq, Jdq = pinocchio.dIntegrate(self.model, q, dq)
        if firstsecond == 'first':
            # Derivative wrt x
            return block_diag(np.asarray(Jq), np.eye(self.model.nv))
        else:
            # Derivative wrt v
            return block_diag(np.asarray(Jdq), np.eye(self.model.nv))
