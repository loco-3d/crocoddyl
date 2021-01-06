///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/impulse-friction-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseFrictionCone() {
  bp::register_ptr_to_python<boost::shared_ptr<CostModelImpulseFrictionCone> >();

  bp::class_<CostModelImpulseFrictionCone, bp::bases<CostModelAbstract> >(
      "CostModelImpulseFrictionCone",
      "This cost function defines a residual vector as r = A*f, where A, f describe the linearized friction cone and "
      "the spatial impulse, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameFrictionCone>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the impulse friction cone cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: frame friction cone"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameFrictionCone>(
          bp::args("self", "state", "fref"),
          "Initialize the impulse friction cone cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          ":param state: state of the multibody system\n"
          ":param fref: frame friction cone"))
      .def<void (CostModelImpulseFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelImpulseFrictionCone::calc, bp::args("self", "data", "x", "u"),
          "Compute the impulse friction cost.\n\n"
          ":param data: cost data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input")
      .def<void (CostModelImpulseFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelImpulseFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelImpulseFrictionCone::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the impulse friction cone cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n")
      .def<void (CostModelImpulseFrictionCone::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                  const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelImpulseFrictionCone::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the impulse friction cone cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelImpulseFrictionCone::get_reference<FrameFrictionCone>,
                    &CostModelImpulseFrictionCone::set_reference<FrameFrictionCone>, "reference frame friction cone");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataImpulseFrictionCone> >();

  bp::class_<CostDataImpulseFrictionCone, bp::bases<CostDataAbstract> >(
      "CostDataImpulseFrictionCone", "Data for impulse friction cone cost.\n\n",
      bp::init<CostModelImpulseFrictionCone*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create impulse friction cone cost data.\n\n"
          ":param model: impulse friction cone cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "impulse",
          bp::make_getter(&CostDataImpulseFrictionCone::impulse, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataImpulseFrictionCone::impulse), "impulse data associated with the current cost");
}

}  // namespace python
}  // namespace crocoddyl
