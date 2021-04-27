///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/costs/residual.hpp"

namespace crocoddyl {
namespace python {

void exposeCostResidual() {
  bp::register_ptr_to_python<boost::shared_ptr<CostModelResidual> >();

  bp::class_<CostModelResidual, bp::bases<CostModelAbstract> >(
      "CostModelResidual",
      "This cost function uses a residual vector with a Gauss-Newton assumption to define a cost term.",
      bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ActivationModelAbstract>,
               boost::shared_ptr<ResidualModelAbstract> >(bp::args("self", "state", "activation", "residual"),
                                                          "Initialize the residual cost model.\n\n"
                                                          ":param state: state description\n"
                                                          ":param activation: activation model\n"
                                                          ":param residual: residual model"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, boost::shared_ptr<ResidualModelAbstract> >(
          bp::args("self", "state", "residual"),
          "Initialize the residual cost model.\n\n"
          ":param state: state description\n"
          ":param residual: residual model"))
      .def<void (CostModelResidual::*)(const boost::shared_ptr<CostDataAbstract>&,
                                       const Eigen::Ref<const Eigen::VectorXd>&,
                                       const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelResidual::calc, bp::args("self", "data", "x", "u"),
          "Compute the residual cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelResidual::*)(const boost::shared_ptr<CostDataAbstract>&,
                                       const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelAbstract::calc,
                                                                                  bp::args("self", "data", "x"))
      .def<void (CostModelResidual::*)(const boost::shared_ptr<CostDataAbstract>&,
                                       const Eigen::Ref<const Eigen::VectorXd>&,
                                       const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelResidual::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the residual cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelResidual::*)(const boost::shared_ptr<CostDataAbstract>&,
                                       const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelResidual::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the residual cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataResidual> >();

  bp::class_<CostDataResidual, bp::bases<CostDataAbstract> >(
      "CostDataResidual", "Data for residual cost.\n\n",
      bp::init<CostModelResidual*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create residual cost data.\n\n"
          ":param model: residual cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("Arr_Rx", bp::make_getter(&CostDataResidual::Arr_Rx, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Rx (deriv of residue)")
      .add_property("Arr_Ru", bp::make_getter(&CostDataResidual::Arr_Ru, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Ru (deriv of residue)");
}

}  // namespace python
}  // namespace crocoddyl
