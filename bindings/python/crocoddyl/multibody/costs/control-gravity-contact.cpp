///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/control-gravity-contact.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostControlGravContact() {
  bp::class_<CostModelControlGravContact, bp::bases<CostModelAbstract> >(
      "CostModelControlGravContact",
      "This cost function defines a residual vector as r = u - computeStaticTorque(q,fext), with u and q,fext as the control and position, external forces ",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the control cost model.\n\n"
          "The default reference control is obtained from np.zero(nu), with nu obtained from activation.nr.\n"
          ":param state: state description\n"
          ":param activation: activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, int>(
          bp::args("self", "state", "activation", "nu"),
          "Initialize the control cost model.\n\n"
          "The default reference control is obtained from np.zero(nu).\n"
          ":param state: state description\n"
          ":param activation: activation model\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>>(
          bp::args("self", "state"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2). The default reference "
          "control is obtained from np.zero(state.nv).\n"
          ":param state: state description"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int>(
          bp::args("self", "state", "nu"),
          "Initialize the control cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2). The default reference "
          "control is obtained from np.zero(nu)\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def<void (CostModelControlGravContact::*)(const boost::shared_ptr<CostDataAbstract>&,
                                      const Eigen::Ref<const Eigen::VectorXd>&,
                                      const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelControlGravContact::calc, bp::args("self", "data", "x", "u"),
          "Compute the control cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelControlGravContact::*)(const boost::shared_ptr<CostDataAbstract>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelAbstract::calc,
                                                                                     bp::args("self", "data", "x"))
      .def<void (CostModelControlGravContact::*)(const boost::shared_ptr<CostDataAbstract>&,
                                      const Eigen::Ref<const Eigen::VectorXd>&,
                                      const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelControlGravContact::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the control cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelControlGravContact::*)(const boost::shared_ptr<CostDataAbstract>&,
                                          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelControlGravContact::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the control cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.");
   bp::class_<CostDataControlGravContact, bp::bases<CostDataAbstract> >(
      "CostDataControlGravContact", "Data for control gravity cost in contact.\n\n",
      bp::init<CostModelControlGravContact*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create control gravity contact cost data in contact.\n\n"
          ":param model: control gravity cost model in contact\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("dg_dx", bp::make_getter(&CostDataControlGravContact::dg_dx, bp::return_internal_reference<>()),
                    "Partial derivative of rnea in contact with respect to x")
      .add_property("dg_da", bp::make_getter(&CostDataControlGravContact::dg_da, bp::return_internal_reference<>()),
                    "Partial derivative of rnea in contact with respect to a");
}

}  // namespace python
}  // namespace crocoddyl
