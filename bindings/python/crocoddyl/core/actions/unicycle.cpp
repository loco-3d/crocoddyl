///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/unicycle.hpp"

#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Model>
struct ActionModelUnicycleVisitor
    : public bp::def_visitor<ActionModelUnicycleVisitor<Model>> {
  typedef typename Model::ActionDataAbstract Data;
  typedef typename Model::VectorXs VectorXs;
  BOOST_PYTHON_FUNCTION_OVERLOADS(ActionModelLQR_Random_wrap, Model::Random, 2,
                                  4)
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("calc",
           static_cast<void (Model::*)(
               const std::shared_ptr<Data>&, const Eigen::Ref<const VectorXs>&,
               const Eigen::Ref<const VectorXs>&)>(&Model::calc),
           bp::args("self", "data", "x", "u"),
           "Compute the next state and cost value.\n\n"
           "It describes the time-discrete evolution of the unicycle system. "
           "Additionally it computes the cost value associated to this "
           "discrete state and control pair.\n"
           ":param data: action data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
        .def("calc",
             static_cast<void (Model::*)(const std::shared_ptr<Data>&,
                                         const Eigen::Ref<const VectorXs>&)>(
                 &Model::calc),
             bp::args("self", "data", "x"))
        .def(
            "calcDiff",
            static_cast<void (Model::*)(
                const std::shared_ptr<Data>&, const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&)>(&Model::calcDiff),
            bp::args("self", "data", "x", "u"),
            "Compute the derivatives of the unicycle dynamics and cost "
            "functions.\n\n"
            "It computes the partial derivatives of the unicycle system and "
            "the cost function. It assumes that calc has been run first. This "
            "function builds a quadratic approximation of the action model "
            "(i.e. dynamical system and cost function).\n"
            ":param data: action data\n"
            ":param x: state point (dim. state.nx)\n"
            ":param u: control input (dim. nu)")
        .def("calcDiff",
             static_cast<void (Model::*)(const std::shared_ptr<Data>&,
                                         const Eigen::Ref<const VectorXs>&)>(
                 &Model::calcDiff),
             bp::args("self", "data", "x"))
        .def("createData", &Model::createData, bp::args("self"),
             "Create the unicycle action data.")
        .add_property("dt", bp::make_function(&Model::get_dt),
                      bp::make_function(&Model::set_dt), "integration time")
        .add_property("costWeights",
                      bp::make_function(&Model::get_cost_weights,
                                        bp::return_internal_reference<>()),
                      bp::make_function(&Model::set_cost_weights),
                      "cost weights");
  }
};

#define CROCODDYL_ACTION_MODEL_UNICYCLE_PYTHON_BINDINGS(Scalar)               \
  typedef ActionModelUnicycleTpl<Scalar> Model;                               \
  typedef ActionModelAbstractTpl<Scalar> ModelBase;                           \
  bp::register_ptr_to_python<std::shared_ptr<Model>>();                       \
  bp::class_<Model, bp::bases<ModelBase>>(                                    \
      "ActionModelUnicycle",                                                  \
      "Unicycle action model.\n\n"                                            \
      "The transition model of an unicycle system is described as\n"          \
      "    xnext = [v*cos(theta); v*sin(theta); w],\n"                        \
      "where the position is defined by (x, y, theta) and the control input " \
      "by (v,w). Note that the state is defined only with the position. On "  \
      "the other hand, we define the quadratic cost functions for the state " \
      "and control.",                                                         \
      bp::init<>(bp::args("self"), "Initialize the unicycle action model."))  \
      .def(ActionModelUnicycleVisitor<Model>())                               \
      .def(CastVisitor<Model>())                                              \
      .def(PrintableVisitor<Model>())                                         \
      .def(CopyableVisitor<Model>());

#define CROCODDYL_ACTION_DATA_UNICYCLE_PYTHON_BINDINGS(Scalar)               \
  typedef ActionDataUnicycleTpl<Scalar> Data;                                \
  typedef ActionDataAbstractTpl<Scalar> DataBase;                            \
  typedef ActionModelUnicycleTpl<Scalar> Model;                              \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                       \
  bp::class_<Data, bp::bases<DataBase>>(                                     \
      "ActionDataUnicycle",                                                  \
      "Action data for the Unicycle system.\n\n"                             \
      "The unicycle data, apart of common one, contains the cost residuals " \
      "used for the computation of calc and calcDiff.",                      \
      bp::init<Model*>(bp::args("self", "model"),                            \
                       "Create unicycle data.\n\n"                           \
                       ":param model: unicycle action model"))               \
      .def(CopyableVisitor<Data>());

void exposeActionUnicycle() {
  CROCODDYL_PYTHON_SCALARS(CROCODDYL_ACTION_MODEL_UNICYCLE_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_SCALARS(CROCODDYL_ACTION_DATA_UNICYCLE_PYTHON_BINDINGS)
}

}  // namespace python
}  // namespace crocoddyl
