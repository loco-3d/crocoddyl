///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/codegen/cppadcg.hpp>

#include "crocoddyl/core/codegen/action-base.hpp"
#include "crocoddyl/multibody/codegen/pinocchio_cast.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Model>
struct ActionModelCodeGenVisitor
    : public bp::def_visitor<ActionModelCodeGenVisitor<Model>> {
  typedef typename Model::ActionDataAbstract Data;
  typedef typename Model::ADBase ADModel;
  typedef typename Model::Base ModelBase;
  typedef typename Model::VectorXs VectorXs;
  typedef typename Model::ParamsEnvironment ParamsEnvironment;
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl
        .def(bp::init<std::shared_ptr<ADModel>, std::string,
                      bp::optional<std::size_t, ParamsEnvironment, CompilerType,
                                   std::string>>(
            bp::args("self", "ad_model", "lib_fname", "updateParams",
                     "compiler", "compile_options"),
            "Initialize the action codegen action model.\n\n"
            ":param ad_model: action model used to code generate\n"
            ":param lib_fname: name of the code generated library\n"
            ":param updateParams: function used to update the calc and "
            "calcDiff's parameters (default empty function)\n"
            ":param compiler: type of compiler GCC or CLANG (default: CLANG)\n"
            ":param compile_options: Compilation flags (default: '-Ofast "
            "-march=native')"))
        .def(
            "calc",
            static_cast<void (Model::*)(
                const std::shared_ptr<Data>&, const Eigen::Ref<const VectorXs>&,
                const Eigen::Ref<const VectorXs>&)>(&Model::calc),
            bp::args("self", "data", "x", "u"),
            "Compute the next state and cost value from a code-generated "
            "library.\n\n"
            ":param data: action data\n"
            ":param x: time-continuous state vector\n"
            ":param u: time-continuous control input")
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
            "Compute the derivatives of the dynamics and cost functions from a "
            "code-generated library\n\n"
            "In contrast to action models, this code-generated calcDiff "
            "doesn't assumes that `calc()` has been run first. This function "
            "builds a linear-quadratic approximation of the action model (i.e. "
            "dynamical system and cost function).\n"
            ":param data: action data\n"
            ":param x: time-continuous state vector\n"
            ":param u: time-continuous control input\n")
        .def("calcDiff",
             static_cast<void (Model::*)(const std::shared_ptr<Data>&,
                                         const Eigen::Ref<const VectorXs>&)>(
                 &Model::calcDiff),
             bp::args("self", "data", "x"))
        .def("createData", &Model::createData, bp::args("self"),
             "Create the action codegen data.")
        .def("initLib", &Model::initLib, bp::args("self"),
             "initialize the code-generated library")
        .def("compileLib", &Model::compileLib, bp::args("self"),
             "compile the code-generated library")
        .def("existLib", &Model::existLib, bp::args("self"),
             "check if the code-generated library exists\n\n"
             ":return True if the code-generated library exists, otherwise "
             "false.")
        .def("loadLib", &Model::loadLib, bp::args("self", "generate_if_exist"),
             "load the code-generated library\n\n"
             ":param generate_if_exist: true for compiling the library when it "
             "exists (default True)")
        .add_property("nX", bp::make_function(&Model::get_nX),
                      "dimension of the dependent vector used by calc and "
                      "calcDiff functions")
        .add_property(
            "nY1", bp::make_function(&Model::get_nY1),
            "dimension of the independent vector used by calc function")
        .add_property(
            "nY2", bp::make_function(&Model::get_nY2),
            "dimension of the independent vector used by calcDiff function");
  }
};

template <typename Data>
struct ActionDataCodeGeneVisitor
    : public bp::def_visitor<ActionDataCodeGeneVisitor<Data>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.add_property(
          "X", bp::make_getter(&Data::X, bp::return_internal_reference<>()),
          "independent variables used by calc and calcDiff function")
        .add_property(
            "Y1", bp::make_getter(&Data::Y1, bp::return_internal_reference<>()),
            "dependent variables used by calc functione")
        .add_property(
            "Y2", bp::make_getter(&Data::Y2, bp::return_internal_reference<>()),
            "dependent variables used by calcDiff functione")
        .add_property(
            "action",
            bp::make_getter(&Data::action,
                            bp::return_value_policy<bp::return_by_value>()));
  }
};

#define CROCODDYL_ACTION_MODEL_CODEGEN_PYTHON_BINDINGS(Scalar)                 \
  typedef ActionModelCodeGenTpl<Scalar> Model;                                 \
  typedef ActionModelAbstractTpl<Scalar> ModelBase;                            \
  typedef typename Model::ParamsEnvironment ParamsEnvironment;                 \
  bp::register_ptr_to_python<std::shared_ptr<Model>>();                        \
  bp::class_<Model, bp::bases<ModelBase>>(                                     \
      "ActionModelCodeGen",                                                    \
      "Action model used to code generate an action model.\n\n"                \
      "This class relies on CppADCodeGen to generate source code and library " \
      "for a defined action model. To handle this easily, it contains "        \
      "functions for initializing, compiling and loading the library. "        \
      "Additionally, it is possible to configure the compiler and its flags.", \
      bp::init<std::shared_ptr<ModelBase>, std::string,                        \
               bp::optional<std::size_t, ParamsEnvironment, CompilerType,      \
                            std::string>>(                                     \
          bp::args("self", "model", "lib_fname", "updateParams", "compiler",   \
                   "compile_options"),                                         \
          "Initialize the action codegen action model.\n\n"                    \
          ":param model: action model model which we want to code generate\n"  \
          ":param lib_fname: name of the code generated library\n"             \
          ":param updateParams: function used to update the calc and "         \
          "calcDiff's parameters (default empty function)\n"                   \
          ":param compiler: type of compiler GCC or CLANG (default: CLANG)\n"  \
          ":param compile_options: Compilation flags (default: '-Ofast "       \
          "-march=native')"))                                                  \
      .def(ActionModelCodeGenVisitor<Model>())                                 \
      .def(PrintableVisitor<Model>())                                          \
      .def(CopyableVisitor<Model>());

#define CROCODDYL_ACTION_DATA_CODEGEN_PYTHON_BINDINGS(Scalar)           \
  typedef ActionDataCodeGenTpl<Scalar> Data;                            \
  typedef ActionDataAbstractTpl<Scalar> DataBase;                       \
  typedef ActionModelCodeGenTpl<Scalar> Model;                          \
  bp::register_ptr_to_python<std::shared_ptr<Data>>();                  \
  bp::class_<Data, bp::bases<DataBase>>(                                \
      "ActionDataCodeGen", "Action data for the codegen action model.", \
      bp::init<Model*>(bp::args("self", "model"),                       \
                       "Create codegen action data.\n\n"                \
                       ":param model: codegen action model"))           \
      .def(ActionDataCodeGeneVisitor<Data>())                           \
      .def(CopyableVisitor<Data>());

void exposeActionCodeGen() {
  bp::enum_<CompilerType>("CompilerType")
      .value("GCC", CompilerType::GCC)
      .value("CLANG", CompilerType::CLANG);

  CROCODDYL_PYTHON_FLOATINGPOINT_SCALARS(
      CROCODDYL_ACTION_MODEL_CODEGEN_PYTHON_BINDINGS)
  CROCODDYL_PYTHON_FLOATINGPOINT_SCALARS(
      CROCODDYL_ACTION_DATA_CODEGEN_PYTHON_BINDINGS)
}

}  // namespace python
}  // namespace crocoddyl
