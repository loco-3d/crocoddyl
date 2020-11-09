///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/impulse-cop-position.hpp"

namespace crocoddyl {
namespace python {

void exposeCostImpulseCoPPosition() {
  bp::class_<CostModelImpulseCoPPosition, bp::bases<CostModelAbstract> >(
      "CostModelImpulseCoPPosition",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameCoPSupport>(
          bp::args("self", "state", "activation", "cop_support"),
          "Initialize the impulse CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model (default ActivationModelQuadraticBarrier)\n"
          ":param cop_support: impulse frame Id and cop support region"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameCoPSupport>(
          bp::args("self", "state", "cop_support"),
          "Initialize the impulse CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param cop_support: impulse frame Id and cop support region"))
      .def<void (CostModelImpulseCoPPosition::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelImpulseCoPPosition::calc, bp::args("self", "data", "x", "u"),
          "Compute the impulse CoP position cost.\n\n"
          ":param data: cost data\n"
          ":param x: state point\n"
          ":param u: control input")
      .def<void (CostModelImpulseCoPPosition::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelImpulseCoPPosition::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the impulse CoP position cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point\n"
          ":param u: control input\n")
      .def("createData", &CostModelImpulseCoPPosition::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the impulse CoP position cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelImpulseCoPPosition::get_reference<FrameCoPSupport>,
                    &CostModelImpulseCoPPosition::set_reference<FrameCoPSupport>, "reference foot geometry and index")
      .add_property("reference_box", &CostModelImpulseCoPPosition::get_reference<MathBaseTpl<double>::Vector2s>,
                    &CostModelImpulseCoPPosition::set_reference<MathBaseTpl<double>::Vector2s>,
                    "reference foot geometry")
      .add_property("reference_id", &CostModelImpulseCoPPosition::get_reference<FrameIndex>,
                    &CostModelImpulseCoPPosition::set_reference<FrameIndex>, "reference index");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataImpulseCoPPosition> >();

  bp::class_<CostDataImpulseCoPPosition, bp::bases<CostDataAbstract> >(
      "CostDataImpulseCoPPosition", "Data for impulse CoP position cost.\n\n",
      bp::init<CostModelImpulseCoPPosition*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create impulse CoP position cost data.\n\n"
          ":param model: impulse CoP position cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("Arr_Ru", bp::make_getter(&CostDataImpulseCoPPosition::Arr_Ru, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Ru (deriv of residue)")
      .add_property(
          "impulse",
          bp::make_getter(&CostDataImpulseCoPPosition::impulse, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataImpulseCoPPosition::impulse), "impulse data associated with the current cost");
}

}  // namespace python
}  // namespace crocoddyl
