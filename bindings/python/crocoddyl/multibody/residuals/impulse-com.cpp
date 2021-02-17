///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualImpulseCoM() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelImpulseCoM> >();

  bp::class_<ResidualModelImpulseCoM, bp::bases<ResidualModelAbstract> >(
      "ResidualModelImpulseCoM",
      "This residual function defines a residual vector as r = Jcom * (vnext-v), with Jcom as the CoM Jacobian, and\n"
      "vnext the velocity after impact and v the velocity before impact, respectively.",
      bp::init<boost::shared_ptr<StateMultibody> >(bp::args("self", "state"),
                                                   "Initialize the CoM position cost model for impulse dynamics.\n\n"
                                                   "The default nu is obtained from state.nv.\n"
                                                   ":param state: state of the multibody system"))
      .def<void (ResidualModelImpulseCoM::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelImpulseCoM::calc, bp::args("self", "data", "x"),
          "Compute the CoM position residual.\n\n"
          ":param data: residual data\n"
          ":param x: time-discrete state vector")
      .def<void (ResidualModelImpulseCoM::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelImpulseCoM::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelImpulseCoM::calcDiff, bp::args("self", "data", "x"),
          "Compute the derivatives of the CoM position residual for impulse dynamics.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n")
      .def<void (ResidualModelImpulseCoM::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelImpulseCoM::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the CoM position residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for the impulse CoM residual.\n"
           ":param data: shared data\n"
           ":return residual data.");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataImpulseCoM> >();

  bp::class_<ResidualDataImpulseCoM, bp::bases<ResidualDataAbstract> >(
      "ResidualDataImpulseCoM", "Data for impulse CoM residual.\n\n",
      bp::init<ResidualModelImpulseCoM*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact force residual data.\n\n"
          ":param model: impulse CoM residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("impulses",
                    bp::make_getter(&ResidualDataImpulseCoM::impulses, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ResidualDataImpulseCoM::impulses),
                    "impulses data associated with the current residual")
      .add_property("dvc_dq", bp::make_getter(&ResidualDataImpulseCoM::dvc_dq, bp::return_internal_reference<>()),
                    "Jacobian of the CoM velocity")
      .add_property("ddv_dv", bp::make_getter(&ResidualDataImpulseCoM::ddv_dv, bp::return_internal_reference<>()),
                    "Jacobian of the impulse velocity")
      .add_property("pinocchio_internal",
                    bp::make_getter(&ResidualDataImpulseCoM::pinocchio_internal, bp::return_internal_reference<>()),
                    "internal pinocchio data used for extra computations");
}

}  // namespace python
}  // namespace crocoddyl
