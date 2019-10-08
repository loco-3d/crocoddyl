///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_

#include "crocoddyl/multibody/impulse-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ImpulseModelAbstract_wrap : public ImpulseModelAbstract, public bp::wrapper<ImpulseModelAbstract> {
 public:
  ImpulseModelAbstract_wrap(StateMultibody& state, int ni) : ImpulseModelAbstract(state, ni) {}

  void calc(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert(static_cast<std::size_t>(x.size()) == state_.get_nx() && "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x);
  }

  void calcDiff(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true) {
    assert(static_cast<std::size_t>(x.size()) == state_.get_nx() && "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, recalc);
  }

  void updateForce(const boost::shared_ptr<ImpulseDataAbstract>& data, const Eigen::VectorXd& force) {
    assert(static_cast<std::size_t>(force.size()) == ni_ && "force has wrong dimension");
    return bp::call<void>(this->get_override("updateForce").ptr(), data, force);
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ImpulseModel_calcDiff_wraps, ImpulseModelAbstract::calcDiff_wrap, 2, 3)

void exposeImpulseAbstract() {
  bp::class_<ImpulseModelAbstract_wrap, boost::noncopyable>(
      "ImpulseModelAbstract",
      "Abstract impulse model.\n\n"
      "It defines a template for impulse models.\n"
      "The calc and calcDiff functions compute the impulse Jacobian\n"
      "the derivatives respectively.",
      bp::init<StateMultibody&, int>(bp::args(" self", " state", " ni"),
                                     "Initialize the impulse model.\n\n"
                                     ":param state: state of the multibody system\n"
                                     ":param ni: dimension of impulse model")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", pure_virtual(&ImpulseModelAbstract_wrap::calc), bp::args(" self", " data", " x"),
           "Compute the impulse Jacobian\n"
           ":param data: impulse data\n"
           ":param x: state vector")
      .def("calcDiff", pure_virtual(&ImpulseModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " recalc=True"),
           "Compute the derivatives of impulse Jacobian\n"
           ":param data: impulse data\n"
           ":param x: state vector\n"
           ":param recalc: If true, it updates the impulse Jacobian")
      .def("updateForce", pure_virtual(&ImpulseModelAbstract_wrap::updateForce), bp::args(" self", " data", " force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: impulse data\n"
           ":param force: force vector (dimension ni)")
      .def("updateForceDiff", &ImpulseModelAbstract_wrap::updateForceDiff, bp::args(" self", " data", " df_dq"),
           "Update the Jacobian of the impulse force.\n\n"
           "The Jacobian df_dv is zero, then we ignore it\n"
           ":param data: impulse data\n"
           ":param df_dq: Jacobian of the impulse force (dimension ni*nv)")
      .def("createData", &ImpulseModelAbstract_wrap::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the impulse data.\n\n"
           "Each impulse model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined impulse.\n"
           ":param data: Pinocchio data\n"
           ":return impulse data.")
      .add_property("state",
                    bp::make_function(&ImpulseModelAbstract_wrap::get_state, bp::return_internal_reference<>()),
                    "state of the multibody system")
      .add_property(
          "ni", bp::make_function(&ImpulseModelAbstract_wrap::get_ni, bp::return_value_policy<bp::return_by_value>()),
          "dimension of impulse");

  bp::class_<ImpulseDataAbstract, boost::shared_ptr<ImpulseDataAbstract>, boost::noncopyable>(
      "ImpulseDataAbstract", "Abstract class for impulse datas.\n\n",
      bp::init<ImpulseModelAbstract*, pinocchio::Data*>(
          bp::args(" self", " model", " data"),
          "Create common data shared between impulse models.\n\n"
          ":param model: impulse model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 3>()])
      .add_property("pinocchio", bp::make_getter(&ImpulseDataAbstract::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("Jc", bp::make_getter(&ImpulseDataAbstract::Jc, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ImpulseDataAbstract::Jc), "impulse Jacobian")
      .add_property("dv0_dq",
                    bp::make_getter(&ImpulseDataAbstract::dv0_dq, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ImpulseDataAbstract::dv0_dq), "Jacobian of the previous impulse velocity")
      .def_readwrite("joint", &ImpulseDataAbstract::joint, "joint index of the impulse frame")
      .def_readwrite("f", &ImpulseDataAbstract::f, "external spatial forces");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
