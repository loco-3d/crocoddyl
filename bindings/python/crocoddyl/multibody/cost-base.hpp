///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
#define PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_

#include "crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class CostModelAbstract_wrap : public CostModelAbstract, public bp::wrapper<CostModelAbstract> {
 public:
  using CostModelAbstract::ndx_;
  using CostModelAbstract::nq_;
  using CostModelAbstract::nr_;
  using CostModelAbstract::nu_;
  using CostModelAbstract::nv_;
  using CostModelAbstract::nx_;
  using CostModelAbstract::unone_;

  CostModelAbstract_wrap(pinocchio::Model* const model, ActivationModelAbstract* const activation, int nu,
                         bool with_residuals = true)
      : CostModelAbstract(model, activation, nu, with_residuals) {}

  CostModelAbstract_wrap(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                         bool with_residuals = true)
      : CostModelAbstract(model, activation, with_residuals) {}

  CostModelAbstract_wrap(pinocchio::Model* const model, int nr, int nu, bool with_residuals = true)
      : CostModelAbstract(model, nr, nu, with_residuals), bp::wrapper<CostModelAbstract>() {}

  CostModelAbstract_wrap(pinocchio::Model* const model, int nr, bool with_residuals = true)
      : CostModelAbstract(model, nr, with_residuals), bp::wrapper<CostModelAbstract>() {}

  void calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) {
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u, recalc);
  }

  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data) {
    return boost::make_shared<CostDataAbstract>(this, data);
  }
};

void exposeCostMultibody() {
  bp::class_<CostModelAbstract_wrap, boost::noncopyable>(
      "CostModelAbstract",
      "Abstract multibody cost model using Pinocchio.\n\n"
      "It defines a template of cost model whose residual and derivatives can be retrieved from\n"
      "Pinocchio data, through the calc and calcDiff functions, respectively.",
      bp::init<pinocchio::Model*, ActivationModelAbstract*, int, bp::optional<bool> >(
          bp::args(" self", " model", " activation", " nu=model.nv", " withResiduals=True"),
          "Initialize the differential action model.\n\n"
          ":param model: Pinocchio model of the multibody system\n"
          ":param activation: Activation model\n"
          ":param nu: dimension of control vector\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<pinocchio::Model*, ActivationModelAbstract*, bp::optional<bool> >(
          bp::args(" self", " model", " activation", " withResiduals=True"),
          "Initialize the differential action model.\n\n"
          ":param model: Pinocchio model of the multibody system\n"
          ":param activation: Activation model\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<pinocchio::Model*, int, int, bp::optional<bool> >(
          bp::args(" self", " model", " nr", " nu=model.nv", " withResiduals=True"),
          "Initialize the differential action model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param model: Pinocchio model of the multibody system\n"
          ":param nr: dimension of cost vector\n"
          ":param nu: dimension of control vector\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<pinocchio::Model*, int, bp::optional<bool> >(
          bp::args(" self", " model", " nr", " withResiduals=True"),
          "Initialize the differential action model.\n\n"
          ":param model: Pinocchio model of the multibody system\n"
          ":param nr: dimension of cost vector\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", pure_virtual(&CostModelAbstract_wrap::calc), bp::args(" self", " data", " x", " u"),
           "Compute the cost value and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", pure_virtual(&CostModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " u", " recalc=True"),
           "Compute the derivatives of the cost function and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input\n"
           ":param recalc: If true, it updates the cost value.")
      .def("createData", &CostModelAbstract_wrap::createData, bp::args(" self"),
           "Create the cost data.\n\n"
           "Each cost model has its own data  that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":return cost data.")
      .add_property("pinocchio",
                    bp::make_function(&CostModelAbstract_wrap::get_pinocchio,
                                      bp::return_value_policy<bp::reference_existing_object>()),
                    "pinocchio model")
      .add_property("activation",
                    bp::make_function(&CostModelAbstract_wrap::get_activation,
                                      bp::return_value_policy<bp::reference_existing_object>()),
                    "activation model")
      .add_property("nq", &CostModelAbstract_wrap::nq_, "dimension of configuration vector")
      .add_property("nv", &CostModelAbstract_wrap::nv_, "dimension of velocity vector")
      .add_property("nu", &CostModelAbstract_wrap::nu_, "dimension of control vector")
      .add_property("nx", &CostModelAbstract_wrap::nx_, "dimension of state configuration vector")
      .add_property("ndx", &CostModelAbstract_wrap::ndx_, "dimension of state tangent vector")
      .add_property("nr", &CostModelAbstract_wrap::nr_, "dimension of cost-residual vector")
      .add_property("unone",
                    bp::make_getter(&CostModelAbstract_wrap::unone_, bp::return_value_policy<bp::return_by_value>()),
                    "default control vector");

  bp::class_<CostDataAbstract, boost::shared_ptr<CostDataAbstract>, boost::noncopyable>(
      "CostDataAbstract", "Abstract class for cost datas.\n\n",
      bp::init<CostModelAbstract*, pinocchio::Data*>(bp::args(" self", " model"),
                                                     "Create common data shared between cost models.\n\n"))
      .add_property("pinocchio",
                    bp::make_function(&CostDataAbstract::get_pinocchio,
                                      bp::return_value_policy<bp::reference_existing_object>()),
                    "pinocchio data")
      .add_property("cost", bp::make_getter(&CostDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::cost), "cost value")
      .add_property("Lx", bp::make_getter(&CostDataAbstract::Lx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lx), "Jacobian of the cost")
      .add_property("Lu", bp::make_getter(&CostDataAbstract::Lu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lu), "Jacobian of the cost")
      .add_property("Lxx", bp::make_getter(&CostDataAbstract::Lxx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lxx), "Hessian of the cost")
      .add_property("Lxu", bp::make_getter(&CostDataAbstract::Lxu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Lxu), "Hessian of the cost")
      .add_property("Luu", bp::make_getter(&CostDataAbstract::Luu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Luu), "Hessian of the cost")
      .add_property("costResiduals",
                    bp::make_getter(&CostDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::r))
      .add_property("Rx", bp::make_getter(&CostDataAbstract::Rx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Rx))
      .add_property("Ru", bp::make_getter(&CostDataAbstract::Ru, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::Ru));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_