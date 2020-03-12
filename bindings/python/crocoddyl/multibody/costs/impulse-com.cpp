///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseCoM() {
  bp::class_<CostModelImpulseCoM, bp::bases<CostModelAbstract> >(
      "CostModelImpulseCoM",
      "Penalize the impulse on the CoM as the sum-of-square of ||Jcom * (vnext-v)||,\n"
      "with vnext the velocity after impact and v the velocity before impact.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the CoM position cost model for impulse dynamics.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(
          bp::args("self", "state"),
          "Initialize the CoM position cost model for impulse dynamics.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(3), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system"))
      .def("calc", &CostModelImpulseCoM::calc_wrap, bp::args("self", "data", "x"),
           "Compute the CoM position cost.\n\n"
           ":param data: cost data\n"
           ":param x: time-discrete state vector")
      .def<void (CostModelImpulseCoM::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelImpulseCoM::calcDiff_wrap, bp::args("self", "data", "x"),
          "Compute the derivatives of the CoM position cost for impulse dynamics.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n")
      .def("createData", &CostModelImpulseCoM::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the CoM position cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataImpulseCoM> >();

  bp::class_<CostDataImpulseCoM, bp::bases<CostDataAbstract> >(
      "CostDataImpulseCoM", "Data for impulse CoM cost.\n\n",
      bp::init<CostModelImpulseCoM*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact force cost data.\n\n"
          ":param model: impulse CoM cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("impulses",
                    bp::make_getter(&CostDataImpulseCoM::impulses, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataImpulseCoM::impulses), "impulses data associated with the current cost")
      .add_property("Arr_Rx", bp::make_getter(&CostDataImpulseCoM::Arr_Rx, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Rx (deriv of residue)")
      .add_property("dvc_dq", bp::make_getter(&CostDataImpulseCoM::dvc_dq, bp::return_internal_reference<>()),
                    "Jacobian of the CoM velocity")
      .add_property("ddv_dv", bp::make_getter(&CostDataImpulseCoM::ddv_dv, bp::return_internal_reference<>()),
                    "Jacobian of the impulse velocity")
      .add_property("pinocchio_internal",
                    bp::make_getter(&CostDataImpulseCoM::pinocchio_internal, bp::return_internal_reference<>()),
                    "internal pinocchio data used for extra computations");
}

}  // namespace python
}  // namespace crocoddyl
