from collections import OrderedDict

import numpy as np
import pinocchio
from pinocchio.utils import cross, skew
from utils import a2m, m2a


class ContactModelPinocchio:
    def __init__(self, pinocchioModel, ncontact, nu=None):
        assert (hasattr(self, 'ContactDataType'))
        self.pinocchio = pinocchioModel
        self.nq, self.nv = pinocchioModel.nq, pinocchioModel.nv
        self.nx = self.nq + self.nv
        self.ndx = 2 * self.nv
        self.nu = nu
        self.ncontact = ncontact

    def createData(self, pinocchioData):
        return self.ContactDataType(self, pinocchioData)

    def calc(self, data, x):
        assert (False and "This should be defined in the derivative class.")

    def calcDiff(self, data, x, recalc=True):
        assert (False and "This should be defined in the derivative class.")

    def setForces(self, data, forcesArr, forcesVec=None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        If forcesVec is not none, sum the result in it. Otherwise, reset self.fs
        and put the result there.
        '''
        assert (False and "This should be defined in the derivative class.")
        return self.forces

    def setForcesDiff(self, data, df_dx, df_du):
        '''
        Feed back the derivative model from the action model.
        '''
        data.df_dx = df_dx
        data.df_du = df_du
        if self.nu is None:
            self.nu = df_du.shape[1]
        else:
            assert (df_du.shape[1] == self.nu)


class ContactDataPinocchio:
    def __init__(self, model, pinocchioData):
        nc, nv, ndx = model.ncontact, model.nv, model.ndx
        self.pinocchio = pinocchioData
        self.J = np.zeros([nc, nv])
        self.a0 = np.zeros(nc)
        self.Ax = np.zeros([nc, ndx])
        self.Aq = self.Ax[:, :nv]
        self.Av = self.Ax[:, nv:]
        self.f = np.nan  # not set at construction type
        self.forces = pinocchio.StdVect_Force()
        for i in range(model.pinocchio.njoints):
            self.forces.append(pinocchio.Force.Zero())


class ContactModel3D(ContactModelPinocchio):
    def __init__(self, pinocchioModel, frame, ref=None, gains=[0., 0.]):
        self.ContactDataType = ContactData3D
        ContactModelPinocchio.__init__(self, pinocchioModel, ncontact=3)
        self.frame = frame
        self.ref = ref
        self.gains = gains

    def calc(self, data, x):
        # We suppose forwardKinematics(q,v,np.zero), computeJointJacobian and updateFramePlacement already
        # computed.
        assert (self.ref is not None or self.gains[0] == 0.)
        data.v = pinocchio.getFrameVelocity(self.pinocchio, data.pinocchio, self.frame).copy()
        vw = data.v.angular
        vv = data.v.linear

        J6 = pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame, pinocchio.ReferenceFrame.LOCAL)
        data.J[:, :] = J6[:3, :]
        data.Jw[:, :] = J6[3:, :]

        data.a0[:] = (
            pinocchio.getFrameAcceleration(self.pinocchio, data.pinocchio, self.frame).linear + cross(vw, vv)).flat
        if self.gains[0] != 0.:
            data.a0[:] += self.gains[0] * (m2a(data.pinocchio.oMf[self.frame].translation) - self.ref)
        if self.gains[1] != 0.:
            data.a0[:] += self.gains[1] * m2a(vv)

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)
        dv_dq, da_dq, da_dv, da_da = pinocchio.getJointAccelerationDerivatives(
            self.pinocchio, data.pinocchio, data.joint, pinocchio.ReferenceFrame.LOCAL)
        dv_dq, dv_dvq = pinocchio.getJointVelocityDerivatives(self.pinocchio, data.pinocchio, data.joint,
                                                              pinocchio.ReferenceFrame.LOCAL)

        vw = data.v.angular
        vv = data.v.linear

        data.Aq[:, :] = (
            data.fXj * da_dq)[:3, :] + skew(vw) * (data.fXj * dv_dq)[:3, :] - skew(vv) * (data.fXj * dv_dq)[3:, :]
        data.Av[:, :] = (data.fXj * da_dv)[:3, :] + skew(vw) * data.J - skew(vv) * data.Jw
        R = data.pinocchio.oMf[self.frame].rotation

        if self.gains[0] != 0.:
            data.Aq[:, :] += self.gains[0] * R * pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame,
                                                                            pinocchio.ReferenceFrame.LOCAL)[:3, :]
        if self.gains[1] != 0.:
            data.Aq[:, :] += self.gains[1] * (data.fXj[:3, :] * dv_dq)
            data.Av[:, :] += self.gains[1] * (data.fXj[:3, :] * dv_dvq)

    def setForces(self, data, forcesArr, forcesVec=None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        '''
        # In the dynamic equation, we wrote M*a + J.T*fdyn, while in the ABA it would be
        # M*a + b = tau + J.T faba, so faba = -fdyn (note the minus operator before a2m).
        data.f = forcesArr
        if forcesVec is None:
            forcesVec = data.forces
            data.forces[data.joint] *= 0
        forcesVec[data.joint] += data.jMf * pinocchio.Force(a2m(forcesArr), np.zeros((3, 1)))
        return forcesVec


class ContactData3D(ContactDataPinocchio):
    def __init__(self, model, pinocchioData):
        ContactDataPinocchio.__init__(self, model, pinocchioData)
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action
        self.v = None
        self.vv = np.zeros([3, 1])
        self.vw = np.zeros([3, 1])
        self.Jw = np.zeros([3, model.nv])


class ContactModel6D(ContactModelPinocchio):
    def __init__(self, pinocchioModel, frame, ref=None, gains=[0., 0.]):
        self.ContactDataType = ContactData6D
        ContactModelPinocchio.__init__(self, pinocchioModel, ncontact=6)
        self.frame = frame
        self.ref = ref
        self.gains = gains

    def calc(self, data, x):
        # We suppose forwardKinematics(q,v,a), computeJointJacobian and updateFramePlacement already
        # computed.
        assert (self.ref is not None or self.gains[0] == 0.)
        data.J[:, :] = pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame,
                                                  pinocchio.ReferenceFrame.LOCAL)
        data.a0[:] = pinocchio.getFrameAcceleration(self.pinocchio, data.pinocchio, self.frame).vector.flat
        if self.gains[0] != 0.:
            data.rMf = self.ref.inverse() * data.pinocchio.oMf[self.frame]
            data.a0[:] += self.gains[0] * m2a(pinocchio.log(data.rMf).vector)
        if self.gains[1] != 0.:
            data.a0[:] += self.gains[1] * m2a(
                pinocchio.getFrameVelocity(self.pinocchio, data.pinocchio, self.frame).vector)

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)

        dv_dq, da_dq, da_dv, da_da = pinocchio.getJointAccelerationDerivatives(
            self.pinocchio, data.pinocchio, data.joint, pinocchio.ReferenceFrame.LOCAL)
        dv_dq, dv_dvq = pinocchio.getJointVelocityDerivatives(self.pinocchio, data.pinocchio, data.joint,
                                                              pinocchio.ReferenceFrame.LOCAL)

        data.Aq[:, :] = data.fXj * da_dq
        data.Av[:, :] = data.fXj * da_dv

        if self.gains[0] != 0.:
            data.Aq[:, :] += self.gains[0] * np.dot(
                pinocchio.Jlog6(data.rMf),
                pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame, pinocchio.ReferenceFrame.LOCAL))
        if self.gains[1] != 0.:
            data.Aq[:, :] += self.gains[1] * data.fXj * dv_dq
            data.Av[:, :] += self.gains[1] * data.fXj * dv_dvq

    def setForces(self, data, forcesArr, forcesVec=None):
        '''
        Convert a numpy array of forces into a stdVector of spatial forces.
        Side effect: keep the force values in data.
        '''
        # In the dynamic equation, we wrote M*a + J.T*fdyn, while in the ABA it would be
        # M*a + b = tau + J.T faba, so faba = -fdyn (note the minus operator before a2m).
        data.f = forcesArr
        if forcesVec is None:
            forcesVec = data.forces
            data.forces[data.joint] *= 0
        forcesVec[data.joint] += data.jMf * pinocchio.Force(a2m(forcesArr))
        return forcesVec


class ContactData6D(ContactDataPinocchio):
    def __init__(self, model, pinocchioData):
        ContactDataPinocchio.__init__(self, model, pinocchioData)
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action
        self.rMf = None


class ContactModelMultiple(ContactModelPinocchio):
    def __init__(self, pinocchioModel):
        self.ContactDataType = ContactDataMultiple
        ContactModelPinocchio.__init__(self, pinocchioModel, ncontact=0)
        self.contacts = OrderedDict()

    def addContact(self, name, contact):
        self.contacts.update([[name, contact]])
        self.ncontact += contact.ncontact

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.contacts[key]
        elif isinstance(key, ContactModelPinocchio):
            filter = [v for k, v in self.contacts.items() if v.contact == key]
            assert (len(filter) == 1 and "The given key is not or not unique in the contact dict. ")
            return filter[0]
        else:
            raise (KeyError("The key should be string or contactmodel."))

    def calc(self, data, x):
        npast = 0
        for m, d in zip(self.contacts.values(), data.contacts.values()):
            m.calc(d, x)
            data.a0[npast:npast + m.ncontact] = d.a0
            data.J[npast:npast + m.ncontact, :] = d.J
            npast += m.ncontact

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)
        npast = 0
        for m, d in zip(self.contacts.values(), data.contacts.values()):
            m.calcDiff(d, x, recalc=False)
            data.Ax[npast:npast + m.ncontact, :] = d.Ax
            npast += m.ncontact

    def setForces(self, data, fsArr):
        npast = 0
        for i, f in enumerate(data.forces):
            data.forces[i] *= 0
        for m, d in zip(self.contacts.values(), data.contacts.values()):
            m.setForces(d, fsArr[npast:npast + m.ncontact], data.forces)
            npast += m.ncontact
        return data.forces

    def setForcesDiff(self, data, df_dx, df_du):
        '''
        Feed back the derivative model from the action model.
        '''
        if self.nu is None:
            self.nu = df_du.shape[1]
        else:
            assert (df_du.shape[1] == self.nu)
        npast = 0
        for m, d in zip(self.contacts.values(), data.contacts.values()):
            m.setForcesDiff(d, df_dx[npast:npast + m.ncontact, :], df_du[npast:npast + m.ncontact, :])
            npast += m.ncontact


class ContactDataMultiple(ContactDataPinocchio):
    def __init__(self, model, pinocchioData):
        ContactDataPinocchio.__init__(self, model, pinocchioData)
        self.model = model
        self.contacts = OrderedDict([[k, m.createData(pinocchioData)] for k, m in model.contacts.items()])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.contacts[key]
        elif isinstance(key, ContactModelPinocchio):
            filter = [k for k, v in self.model.contacts.items() if v == key]
            assert (len(filter) == 1 and "The given key is not or not unique in the contact dict. ")
            return self.contacts[filter[0]]
        else:
            raise (KeyError("The key should be string or contactmodel."))
