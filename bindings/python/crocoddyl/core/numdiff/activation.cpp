///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/numdiff/activation.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationNumDiff() {
  bp::class_<ActivationModelNumDiff, bp::bases<ActivationModelAbstract> >(
      "ActivationModelNumDiff", "Abstract class for computing calcDiff by using numerical differentiation.\n\n",
      bp::init<boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "model"),
          "Initialize the activation model NumDiff.\n\n"
          ":param model: activation model where we compute the derivatives through NumDiff"))
      .def("calc", &ActivationModelNumDiff::calc_wrap, bp::args("self", "data", "r"),
           "Compute the activation value.\n\n"
           "The activation evolution is described in model.\n"
           ":param data: NumDiff action data\n"
           ":param r: residual vector")
      .def<void (ActivationModelNumDiff::*)(const boost::shared_ptr<ActivationDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &ActivationModelNumDiff::calcDiff_wrap, bp::args("self", "data", "r"),
          "Compute the derivatives of the residual.\n\n"
          "It computes the Jacobian and Hessian using numerical differentiation.\n"
          ":param data: NumDiff action data\n"
          ":param r: residual vector\n")
      .def("createData", &ActivationModelNumDiff::createData, bp::args("self"),
           "Create the activation data.\n\n"
           "Each activation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property(
          "model",
          bp::make_function(&ActivationModelNumDiff::get_model, bp::return_value_policy<bp::return_by_value>()),
          "action model")
      .add_property(
          "disturbance",
          bp::make_function(&ActivationModelNumDiff::get_disturbance, bp::return_value_policy<bp::return_by_value>()),
          "disturbance value used in the numerical differentiation");

  bp::register_ptr_to_python<boost::shared_ptr<ActivationDataNumDiff> >();

  bp::class_<ActivationDataNumDiff, bp::bases<ActivationDataAbstract> >(
      "ActivationDataNumDiff", "Numerical differentiation activation data.",
      bp::init<ActivationModelNumDiff*>(bp::args("self", "model"),
                                        "Create numerical differentiation activation data.\n\n"
                                        ":param model: numdiff activation model"))
      .add_property("dr", bp::make_getter(&ActivationDataNumDiff::dr, bp::return_internal_reference<>()),
                    "disturbance.")
      .add_property("rp", bp::make_getter(&ActivationDataNumDiff::rp, bp::return_internal_reference<>()),
                    "input plus the disturbance.")
      .add_property("data_0",
                    bp::make_getter(&ActivationDataNumDiff::data_0, bp::return_value_policy<bp::return_by_value>()),
                    "data that contains the final results")
      .add_property("data_rp",
                    bp::make_getter(&ActivationDataNumDiff::data_rp, bp::return_value_policy<bp::return_by_value>()),
                    "temporary data associated with the input variation")
      .add_property("data_r2p",
                    bp::make_getter(&ActivationDataNumDiff::data_r2p, bp::return_value_policy<bp::return_by_value>()),
                    "temporary data associated with the input variation");
}

}  // namespace python
}  // namespace crocoddyl
