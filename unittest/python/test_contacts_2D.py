import numpy.core.multiarray
import os
import sys

import numpy as np

import pinocchio
from pinocchio.utils import rand,zero
import crocoddyl
import matplotlib.pylab as plt
from numpy.linalg import norm, solve

sys.path.append("/local/src/crocoddyl/unittest/")
from testutils import NUMDIFF_MODIFIER, assertNumDiff, df_dq, df_dx
pinocchio.switchToNumpyMatrix()

def m2a(m): return np.array(m.flat)

def a2m(a): return np.matrix(a).T

EPS = np.finfo(float).eps
NUMDIFF_MODIFIER = 1e4

# Loading Talos arm with FF TODO use a bided or quadruped
# -----------------------------------------------------------------------------
# Load collision model
URDF_FILENAME = "solo.urdf"
SRDF_FILENAME = "solo.srdf"
SRDF_SUBPATH = "/solo_description/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/solo_description/robots/" + URDF_FILENAME

modelPath = "/opt/openrobots/share/example-robot-data/robots"
rmodel, geomModel, visualModel = pinocchio.buildModelsFromUrdf(modelPath + URDF_SUBPATH, modelPath,pinocchio.JointModelFreeFlyer())


rdata = rmodel.createData()
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)


pinocchio.loadReferenceConfigurations(rmodel,modelPath + SRDF_SUBPATH, False)
q = rmodel.referenceConfigurations["standing"]
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv - 6))

pinocchio.forwardKinematics(rmodel, rdata, q, v, zero(rmodel.nv))
pinocchio.computeJointJacobians(rmodel, rdata)
pinocchio.updateFramePlacements(rmodel, rdata)
pinocchio.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, zero(rmodel.nv))

FRcontact = 'FR_FOOT'
FRId = rmodel.getFrameId(FRcontact)
frameFR = crocoddyl.FrameTranslation(FRId, rdata.oMf[FRId].translation)
contactModel = crocoddyl.ContactModel2D(state, frameFR, actuation.nu, np.matrix([1, 4]).T)

contactData = contactModel.createData(rdata)
contactModel.calc(contactData, x)
contactModel.calcDiff(contactData, x)

rdata2 = rmodel.createData()
pinocchio.computeAllTerms(rmodel, rdata2, q, v)
pinocchio.updateFramePlacements(rmodel, rdata2)
contactData2 = contactModel.createData(rdata2)
contactModel.calc(contactData2, x)
assert (norm(contactData.a0 - contactData2.a0) < 1e-9)
assert (norm(contactData.Jc - contactData2.Jc) < 1e-9)


def returna0(q, v):
    x = m2a(np.concatenate([q, v]))
    pinocchio.computeAllTerms(rmodel, rdata2, q, v)
    pinocchio.updateFramePlacements(rmodel, rdata2)
    contactModel.calc(contactData2, x)
    return contactData2.a0.copy()  # .copy()


Aq_numdiff = df_dq(rmodel, lambda _q: returna0(_q, v), q) 
Av_numdiff = df_dx(lambda _v: returna0(q, _v), v) 

assertNumDiff(contactData.da0_dx[:,:rmodel.nv], Aq_numdiff,
              NUMDIFF_MODIFIER * np.sqrt(2 * EPS))  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(contactData.da0_dx[:,rmodel.nv:], Av_numdiff,
              NUMDIFF_MODIFIER * np.sqrt(2 * EPS))  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

Aq_numdiff = df_dq(rmodel, lambda _q: returna0(_q, v), q, h=EPS)
Av_numdiff = df_dx(lambda _v: returna0(q, _v), v, h=EPS)

assert (np.isclose(contactData.da0_dx[:,:rmodel.nv], Aq_numdiff, atol=np.sqrt(EPS)).all())
assert (np.isclose(contactData.da0_dx[:,rmodel.nv:], Av_numdiff, atol=np.sqrt(EPS)).all())


