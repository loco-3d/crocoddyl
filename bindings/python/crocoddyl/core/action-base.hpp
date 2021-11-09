///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class ActionModelAbstract_wrap : public ActionModelAbstract, public bp::wrapper<ActionModelAbstract> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ActionModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, std::size_t nu, std::size_t nr = 1)
      : ActionModelAbstract(state, nu, nr), bp::wrapper<ActionModelAbstract>() {}

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  boost::shared_ptr<ActionDataAbstract> createData() {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<ActionDataAbstract> >(createData.ptr());
    }
    return ActionModelAbstract::createData();
  }

  boost::shared_ptr<ActionDataAbstract> default_createData() { return this->ActionModelAbstract::createData(); }

  void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                   const Eigen::Ref<const Eigen::VectorXd>& x, const std::size_t maxiter, const double tol) {
    if (boost::python::override quasiStatic = this->get_override("quasiStatic")) {
      u = bp::call<Eigen::VectorXd>(quasiStatic.ptr(), data, (Eigen::VectorXd)x, maxiter, tol);
      if (static_cast<std::size_t>(u.size()) != nu_) {
        throw_pretty("Invalid argument: "
                     << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
      }
      return;
    }
    return ActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                           const Eigen::Ref<const Eigen::VectorXd>& x, const std::size_t maxiter, const double tol) {
    return this->ActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }

  void multiplyByFx(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                    Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.cols()) != state_->get_ndx()) {
      throw_pretty("Invalid argument: "
                   << "number of columns of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                          " instead of " + std::to_string(A.cols()));
    }
    if (boost::python::override multiplyByFx = this->get_override("multiplyByFx")) {
      if (A.rows() != out.rows()) {
        throw_pretty("Invalid argument: "
                     << "A and out have different number of rows: " + std::to_string(A.rows()) + " and " +
                            std::to_string(out.rows()));
      }
      if (static_cast<std::size_t>(out.cols()) != state_->get_ndx()) {
        throw_pretty("Invalid argument: "
                     << "number of columns of out is wrong, it should be " + std::to_string(state_->get_ndx()) +
                            " instead of " + std::to_string(out.cols()));
      }
      MatrixXs res = bp::call<MatrixXs>(multiplyByFx.ptr(), (MatrixXs)Fx, (MatrixXs)A);
      if (res.rows() != A.rows() || static_cast<std::size_t>(res.cols()) != state_->get_ndx()) {
        throw_pretty("Invalid argument: "
                     << "resulting matrix has wrong dimension, it should be (" + std::to_string(A.rows()) + "," +
                            std::to_string(state_->get_ndx()) + ") instead of (" + std::to_string(res.rows()) + "," +
                            std::to_string(res.cols()) + ")");
      }
      assigmentOperator(out, res, op);
      return;
    }
    return ActionModelAbstract::multiplyByFx(Fx, A, out, op);
  }

  void multiplyFxTransposeBy(const Eigen::Ref<const MatrixXs>& FxTranspose, const Eigen::Ref<const MatrixXs>& A,
                             Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.rows()) != state_->get_ndx()) {
      throw_pretty("Invalid argument: "
                   << "number of rows of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                          " instead of " + std::to_string(A.rows()));
    }
    if (boost::python::override multiplyFxTransposeBy = this->get_override("multiplyFxTransposeBy")) {
      if (A.cols() != out.cols()) {
        throw_pretty("Invalid argument: "
                     << "A and out have different number of columns: " + std::to_string(A.cols()) + " and " +
                            std::to_string(out.cols()));
      }
      if (static_cast<std::size_t>(out.rows()) != state_->get_ndx()) {
        throw_pretty("Invalid argument: "
                     << "number of rows of out is wrong, it should be " + std::to_string(state_->get_ndx()) +
                            " instead of " + std::to_string(out.cols()));
      }
      MatrixXsRowMajor res =
          bp::call<MatrixXsRowMajor>(multiplyFxTransposeBy.ptr(), (MatrixXs)FxTranspose, (MatrixXs)A);
      if (static_cast<std::size_t>(res.rows()) != state_->get_ndx() || res.cols() != A.cols()) {
        throw_pretty("Invalid argument: "
                     << "resulting matrix has wrong dimension, it should be (" + std::to_string(state_->get_ndx()) +
                            "," + std::to_string(A.cols()) + ") instead of (" + std::to_string(res.rows()) + "," +
                            std::to_string(res.cols()) + ")");
      }
      assigmentOperator(out, res, op);
      return;
    }
    return ActionModelAbstract::multiplyFxTransposeBy(FxTranspose, A, out, op);
  }

  void multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                    Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.cols()) != state_->get_ndx()) {
      throw_pretty("Invalid argument: "
                   << "number of columns of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                          " instead of " + std::to_string(A.cols()));
    }
    if (boost::python::override multiplyByFu = this->get_override("multiplyByFu")) {
      if (A.rows() != out.rows()) {
        throw_pretty("Invalid argument: "
                     << "A and out have different number of rows: " + std::to_string(A.rows()) + " and " +
                            std::to_string(out.rows()));
      }
      if (static_cast<std::size_t>(out.cols()) != nu_) {
        throw_pretty("Invalid argument: "
                     << "number of columns of out is wrong, it should be " + std::to_string(nu_) + " instead of " +
                            std::to_string(out.cols()));
      }
      MatrixXs res = bp::call<MatrixXs>(multiplyByFu.ptr(), (MatrixXs)Fu, (MatrixXs)A);
      if (res.rows() != A.rows() || static_cast<std::size_t>(res.cols()) != nu_) {
        throw_pretty("Invalid argument: "
                     << "resulting matrix has wrong dimension, it should be (" + std::to_string(A.rows()) + "," +
                            std::to_string(nu_) + ") instead of (" + std::to_string(res.rows()) + "," +
                            std::to_string(res.cols()) + ")");
      }
      assigmentOperator(out, res, op);
      return;
    }
    return ActionModelAbstract::multiplyByFu(Fu, A, out, op);
  }

  void multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& FuTranspose, const Eigen::Ref<const MatrixXs>& A,
                             Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.rows()) != state_->get_ndx()) {
      throw_pretty("Invalid argument: "
                   << "number of rows of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                          " instead of " + std::to_string(A.rows()));
    }
    if (boost::python::override multiplyFuTransposeBy = this->get_override("multiplyFuTransposeBy")) {
      if (A.cols() != out.cols()) {
        throw_pretty("Invalid argument: "
                     << "A and out have different number of columns: " + std::to_string(A.cols()) + " and " +
                            std::to_string(out.cols()));
      }
      if (static_cast<std::size_t>(out.rows()) != nu_) {
        throw_pretty("Invalid argument: "
                     << "number of rows of out is wrong, it should be " + std::to_string(nu_) + " instead of " +
                            std::to_string(out.cols()));
      }
      MatrixXsRowMajor res =
          bp::call<MatrixXsRowMajor>(multiplyFuTransposeBy.ptr(), (MatrixXs)FuTranspose, (MatrixXs)A);
      if (static_cast<std::size_t>(res.rows()) != nu_ || res.cols() != A.cols()) {
        throw_pretty("Invalid argument: "
                     << "resulting matrix has wrong dimension, it should be (" + std::to_string(nu_) + "," +
                            std::to_string(A.cols()) + ") instead of (" + std::to_string(res.rows()) + "," +
                            std::to_string(res.cols()) + ")");
      }
      assigmentOperator(out, res, op);
      return;
    }
    return ActionModelAbstract::multiplyFuTransposeBy(FuTranspose, A, out, op);
  }

  void default_multiplyByFx(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    return this->ActionModelAbstract::multiplyByFx(Fx, A, out, op);
  }

  void default_multiplyFxTransposeBy(const Eigen::Ref<const MatrixXs>& FxTranspose,
                                     const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXsRowMajor> out,
                                     const AssignmentOp op) const {
    return this->ActionModelAbstract::multiplyFxTransposeBy(FxTranspose, A, out, op);
  }

  void default_multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    return this->ActionModelAbstract::multiplyByFu(Fu, A, out, op);
  }

  void default_multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& FuTranspose,
                                     const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXsRowMajor> out,
                                     const AssignmentOp op) const {
    return this->ActionModelAbstract::multiplyFuTransposeBy(FuTranspose, A, out, op);
  }

 protected:
  void assigmentOperator(Eigen::Ref<MatrixXs> out, Eigen::Ref<MatrixXs> res, const AssignmentOp op) const {
    assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
    switch (op) {
      case setto: {
        out = res;
        break;
      }
      case addto: {
        out += res;
        break;
      }
      case rmfrom: {
        out -= res;
        break;
      }
      default: {
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }

  void assigmentOperator(Eigen::Ref<MatrixXsRowMajor> out, Eigen::Ref<MatrixXsRowMajor> res,
                         const AssignmentOp op) const {
    assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
    switch (op) {
      case setto: {
        out = res;
        break;
      }
      case addto: {
        out += res;
        break;
      }
      case rmfrom: {
        out -= res;
        break;
      }
      default: {
        throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ActionModel_quasiStatic_wraps, ActionModelAbstract::quasiStatic_x, 2, 4)

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
