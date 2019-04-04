import numpy as np
from crocoddyl import (ActionModelImpact, ActivationModelWeightedQuad, ActuationModelFreeFloating, ContactModel6D,
                       ContactModelMultiple, CostModelCoM, CostModelControl, CostModelForce, CostModelFramePlacement,
                       CostModelFrameVelocity, CostModelImpactCoM, CostModelState, CostModelSum,
                       DifferentialActionModelFloatingInContact, ImpulseModel6D, ImpulseModelMultiple,
                       IntegratedActionModelEuler, StatePinocchio, m2a)
from pinocchio import SE3, Motion


def createMultiphaseShootingProblem(rmodel, rdata, csw, timeStep):
    """
  Create a Multiphase Shooting problem from the output of the centroidal solver.

  :params rmodel: robot model of type pinocchio::model
  :params rdata: robot data of type pinocchio::data
  :params csw: contact sequence wrapper of type ContactSequenceWrapper
  :params timeStep: Scalar timestep between nodes.

  :returns list of IntegratedActionModels
  """

    # -----------------------
    # Define Cost weights
    class Weights:
        com = 1e1
        regx = 1e-3
        regu = 0.
        swing_patch = 1e6
        forces = 0.
        contactv = 1e3
        # Define state cost vector for WeightedActivation
        ff_orientation = 1e1
        xweight = np.array([0] * 3 + [ff_orientation] * 3 + [1.] * (rmodel.nv - 6) + [1.] * rmodel.nv)
        xweight[range(18, 20)] = ff_orientation
        # for patch in swing_patch:  w.swing_patch.append(100.);
        # Define weights for the impact costs.
        imp_state = 1e2
        imp_com = 1e2
        imp_contact_patch = 1e6
        imp_act_com = m2a([0.1, 0.1, 3.0])

        # Define weights for the terminal costs
        term_com = 1e8
        term_regx = 1e4

    w = Weights()
    # ------------------------

    problem_models = []
    actuationff = ActuationModelFreeFloating(rmodel)
    State = StatePinocchio(rmodel)

    active_contact_patch = set()
    active_contact_patch_prev = set()
    for nphase, phase in enumerate(csw.cs.contact_phases):
        t0 = phase.time_trajectory[0]
        t1 = phase.time_trajectory[-1]
        N = int(round((t1 - t0) / timeStep)) + 1
        contact_model = ContactModelMultiple(rmodel)

        # Add contact constraints for the active contact patches.
        # Add SE3 cost for the non-active contact patches.
        swing_patch = []
        active_contact_patch_prev = active_contact_patch.copy()
        active_contact_patch.clear()
        for patch in csw.ee_map.keys():
            if getattr(phase, patch).active:
                active_contact_patch.add(patch)
                active_contact = ContactModel6D(
                    rmodel, frame=rmodel.getFrameId(csw.ee_map[patch]), ref=getattr(phase, patch).placement)
                contact_model.addContact(patch, active_contact)
                # print nphase, "Contact ",patch," added at ", getattr(phase,patch).placement.translation.T
            else:
                swing_patch.append(patch)

        # Check if contact has been added in this phase. If this phase is not zero,
        # add an impulse model to deal with this contact.
        new_contacts = active_contact_patch.difference(active_contact_patch_prev)
        if nphase != 0 and len(new_contacts) != 0:
            # print nphase, "Impact ",[p for p in new_contacts]," added"
            imp_model = ImpulseModelMultiple(
                rmodel, {
                    "Impulse_" + patch: ImpulseModel6D(rmodel, frame=rmodel.getFrameId(csw.ee_map[patch]))
                    for patch in new_contacts
                })
            # Costs for the impulse of a new contact
            cost_model = CostModelSum(rmodel, nu=0)
            # State
            cost_regx = CostModelState(
                rmodel,
                State,
                ref=rmodel.defaultState,
                nu=actuationff.nu,
                activation=ActivationModelWeightedQuad(w.xweight))
            cost_model.addCost("imp_regx", cost_regx, w.imp_state)
            # CoM
            cost_com = CostModelImpactCoM(rmodel, activation=ActivationModelWeightedQuad(w.imp_act_com))
            cost_model.addCost("imp_CoM", cost_com, w.imp_com)
            # Contact Frameplacement
            for patch in new_contacts:
                cost_contact = CostModelFramePlacement(
                    rmodel,
                    frame=rmodel.getFrameId(csw.ee_map[patch]),
                    ref=SE3(np.identity(3), csw.ee_splines[patch].eval(t0)[0]),
                    nu=actuationff.nu)
                cost_model.addCost("imp_contact_" + patch, cost_contact, w.imp_contact_patch)

            imp_action_model = ActionModelImpact(rmodel, imp_model, cost_model)
            problem_models.append(imp_action_model)

        # Define the cost and action models for each timestep in the contact phase.
        # untill [:-1] because in contact sequence timetrajectory, the end-time is
        # also included. e.g., instead of being [0.,0.5], time trajectory is [0,0.5,1.]
        for t in np.linspace(t0, t1, N)[:-1]:
            cost_model = CostModelSum(rmodel, actuationff.nu)

            # For the first node of the phase, add cost v=0 for the contacting foot.
            if t == 0:
                for patch in active_contact_patch:
                    cost_vcontact = CostModelFrameVelocity(
                        rmodel,
                        frame=rmodel.getFrameId(csw.ee_map[patch]),
                        ref=m2a(Motion.Zero().vector),
                        nu=actuationff.nu)
                    cost_model.addCost("contactv_" + patch, cost_vcontact, w.contactv)

            # CoM Cost
            cost_com = CostModelCoM(rmodel, ref=m2a(csw.phi_c.com_vcom.eval(t)[0][:3, :]), nu=actuationff.nu)
            cost_model.addCost("CoM", cost_com, w.com)

            # Forces Cost
            for patch in contact_model.contacts.keys():
                cost_force = CostModelForce(
                    rmodel,
                    contactModel=contact_model.contacts[patch],
                    ref=m2a(csw.phi_c.forces[patch].eval(t)[0]),
                    nu=actuationff.nu)
                cost_model.addCost("forces_" + patch, cost_force, w.forces)
            # Swing patch cost
            for patch in swing_patch:
                cost_swing = CostModelFramePlacement(
                    rmodel,
                    frame=rmodel.getFrameId(csw.ee_map[patch]),
                    ref=SE3(np.identity(3), csw.ee_splines[patch].eval(t)[0]),
                    nu=actuationff.nu)
                cost_model.addCost("swing_" + patch, cost_swing, w.swing_patch)
                # print t, "Swing cost ",patch," added at ", csw.ee_splines[patch].eval(t)[0][:3].T

            # State Regularization
            cost_regx = CostModelState(
                rmodel,
                State,
                ref=rmodel.defaultState,
                nu=actuationff.nu,
                activation=ActivationModelWeightedQuad(w.xweight))
            cost_model.addCost("regx", cost_regx, w.regx)
            # Control Regularization
            cost_regu = CostModelControl(rmodel, nu=actuationff.nu)
            cost_model.addCost("regu", cost_regu, w.regu)

            dmodel = DifferentialActionModelFloatingInContact(rmodel, actuationff, contact_model, cost_model)
            imodel = IntegratedActionModelEuler(dmodel, timeStep=timeStep)
            problem_models.append(imodel)

    # Create Terminal Model.
    contact_model = ContactModelMultiple(rmodel)
    # Add contact constraints for the active contact patches.
    swing_patch = []
    t = t1
    for patch in csw.ee_map.keys():
        if getattr(phase, patch).active:
            active_contact = ContactModel6D(
                rmodel, frame=rmodel.getFrameId(csw.ee_map[patch]), ref=getattr(phase, patch).placement)
            contact_model.addContact(patch, active_contact)
    cost_model = CostModelSum(rmodel, actuationff.nu)
    # CoM Cost
    cost_com = CostModelCoM(rmodel, ref=m2a(csw.phi_c.com_vcom.eval(t)[0][:3, :]), nu=actuationff.nu)
    cost_model.addCost("CoM", cost_com, w.term_com)

    # State Regularization
    cost_regx = CostModelState(
        rmodel, State, ref=rmodel.defaultState, nu=actuationff.nu, activation=ActivationModelWeightedQuad(w.xweight))
    cost_model.addCost("regx", cost_regx, w.term_regx)

    dmodel = DifferentialActionModelFloatingInContact(rmodel, actuationff, contact_model, cost_model)
    imodel = IntegratedActionModelEuler(dmodel)
    problem_models.append(imodel)
    problem_models.append
    return problem_models
