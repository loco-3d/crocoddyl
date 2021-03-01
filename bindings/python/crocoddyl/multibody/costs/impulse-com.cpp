///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseCoM() {  // TODO: Remove once the deprecated update call has been removed in a future
                               // release
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::register_ptr_to_python<boost::shared_ptr<CostModelImpulseCoM> >();

  bp::class_<CostModelImpulseCoM, bp::bases<CostModelAbstract> >(
      "CostModelImpulseCoM",
      "This cost function defines a residual vector as r = Jcom * (vnext-v), with Jcom as the CoM Jacobian, and vnext "
      "the velocity after impact and v the velocity before impact, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the CoM position cost model for impulse dynamics.\n\n"
          "The default nu is obtained from state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(
          bp::args("self", "state"),
          "Initialize the CoM position cost model for impulse dynamics.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system"))
      .def<void (CostModelImpulseCoM::*)(
          const boost::shared_ptr<CostDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelImpulseCoM::calc, bp::args("self", "data", "x"),
                                                     "Compute the CoM position cost.\n\n"
                                                     ":param data: cost data\n"
                                                     ":param x: time-discrete state vector")
      .def<void (CostModelImpulseCoM::*)(const boost::shared_ptr<CostDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelAbstract::calc,
                                                                                    bp::args("self", "data", "x"))
      .def<void (CostModelImpulseCoM::*)(const boost::shared_ptr<CostDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelImpulseCoM::calcDiff, bp::args("self", "data", "x"),
          "Compute the derivatives of the CoM position cost for impulse dynamics.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n")
      .def<void (CostModelImpulseCoM::*)(const boost::shared_ptr<CostDataAbstract>&,
                                         const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
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
      .add_property("Arr_Rx", bp::make_getter(&CostDataImpulseCoM::Arr_Rx, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Rx (deriv of residue)");

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
