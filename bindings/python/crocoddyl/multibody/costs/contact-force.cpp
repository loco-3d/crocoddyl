///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactForce() {
  bp::class_<CostModelContactForce, bp::bases<CostModelAbstract> >(
      "CostModelContactForce",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce, int>(
          bp::args("self", "state", "activation", "fref", "nu"),
          "Initialize the contact force cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference contact force\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact force cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference contact force\n"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce, int>(
          bp::args("self", "state", "fref", "nu"),
          "Initialize the contact force cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference contact force\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce>(
          bp::args("self", "state", "fref"),
          "Initialize the contact force cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6), and nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference force\n"))
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelContactForce::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact force cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelAbstract::calc,
                                                                                      bp::args("self", "data", "x"))
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelContactForce::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the contact force cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelContactForce::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact force cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference",
                    bp::make_function(&CostModelContactForce::get_fref, bp::return_internal_reference<>()),
                    &CostModelContactForce::set_reference<FrameForce>, "reference frame force")
      .add_property("fref", bp::make_function(&CostModelContactForce::get_fref, bp::return_internal_reference<>()),
                    &CostModelContactForce::set_fref, "reference contact force");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataContactForce> >();

  bp::class_<CostDataContactForce, bp::bases<CostDataAbstract> >(
      "CostDataContactForce", "Data for contact force cost.\n\n",
      bp::init<CostModelContactForce*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact force cost data.\n\n"
          ":param model: contact force cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("Arr_Ru", bp::make_getter(&CostDataContactForce::Arr_Ru, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Ru (deriv of residue)")
      .add_property("contact",
                    bp::make_getter(&CostDataContactForce::contact, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataContactForce::contact), "contact data associated with the current cost");
}

}  // namespace python
}  // namespace crocoddyl
