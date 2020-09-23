///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

class DifferentialActionModelAbstract_wrap : public DifferentialActionModelAbstract,
                                             public bp::wrapper<DifferentialActionModelAbstract> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DifferentialActionModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, const std::size_t nu,
                                       const std::size_t nr = 1, const std::size_t ng = 0, const std::size_t nh = 0)
      : DifferentialActionModelAbstract(state, nu, nr, ng, nh), bp::wrapper<DifferentialActionModelAbstract>() {}

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
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

  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
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

  boost::shared_ptr<DifferentialActionDataAbstract> createData() {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<DifferentialActionDataAbstract> >(createData.ptr());
    }
    return DifferentialActionModelAbstract::createData();
  }

  boost::shared_ptr<DifferentialActionDataAbstract> default_createData() {
    return this->DifferentialActionModelAbstract::createData();
  }

  void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                   const Eigen::Ref<const Eigen::VectorXd>& x, const std::size_t maxiter, const double tol) {
    if (boost::python::override quasiStatic = this->get_override("quasiStatic")) {
      u = bp::call<Eigen::VectorXd>(quasiStatic.ptr(), data, (Eigen::VectorXd)x, maxiter, tol);
      if (static_cast<std::size_t>(u.size()) != nu_) {
        throw_pretty("Invalid argument: "
                     << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
      }
      return;
    }
    return DifferentialActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }

  void multiplyByFx(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                    Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.cols()) != state_->get_nv()) {
      throw_pretty("Invalid argument: "
                   << "number of columns of A is wrong, it should be " + std::to_string(state_->get_nv()) +
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
    return DifferentialActionModelAbstract::multiplyByFx(Fx, A, out, op);
  }

  void multiplyFxTransposeBy(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                             Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.rows()) != state_->get_nv()) {
      throw_pretty("Invalid argument: "
                   << "number of rows of A is wrong, it should be " + std::to_string(state_->get_nv()) +
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
      MatrixXsRowMajor res = bp::call<MatrixXsRowMajor>(multiplyFxTransposeBy.ptr(), (MatrixXs)Fx, (MatrixXs)A);
      if (static_cast<std::size_t>(res.rows()) != state_->get_ndx() || res.cols() != A.cols()) {
        throw_pretty("Invalid argument: "
                     << "resulting matrix has wrong dimension, it should be (" + std::to_string(state_->get_ndx()) +
                            "," + std::to_string(A.cols()) + ") instead of (" + std::to_string(res.rows()) + "," +
                            std::to_string(res.cols()) + ")");
      }
      assigmentOperator(out, res, op);
      return;
    }
    return DifferentialActionModelAbstract::multiplyFxTransposeBy(Fx, A, out, op);
  }

  void multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                    Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.cols()) != state_->get_nv()) {
      throw_pretty("Invalid argument: "
                   << "number of columns of A is wrong, it should be " + std::to_string(state_->get_nv()) +
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
    return DifferentialActionModelAbstract::multiplyByFu(Fu, A, out, op);
  }

  void multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                             Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp op) const {
    if (static_cast<std::size_t>(A.rows()) != state_->get_nv()) {
      throw_pretty("Invalid argument: "
                   << "number of rows of A is wrong, it should be " + std::to_string(state_->get_nv()) +
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
      MatrixXsRowMajor res = bp::call<MatrixXsRowMajor>(multiplyFuTransposeBy.ptr(), (MatrixXs)Fu, (MatrixXs)A);
      if (static_cast<std::size_t>(res.rows()) != nu_ || res.cols() != A.cols()) {
        throw_pretty("Invalid argument: "
                     << "resulting matrix has wrong dimension, it should be (" + std::to_string(nu_) + "," +
                            std::to_string(A.cols()) + ") instead of (" + std::to_string(res.rows()) + "," +
                            std::to_string(res.cols()) + ")");
      }
      assigmentOperator(out, res, op);
      return;
    }
    return DifferentialActionModelAbstract::multiplyFuTransposeBy(Fu, A, out, op);
  }

  void default_quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                           Eigen::Ref<Eigen::VectorXd> u, const Eigen::Ref<const Eigen::VectorXd>& x,
                           const std::size_t maxiter, const double tol) {
    return this->DifferentialActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_multiplyByFx(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    return this->DifferentialActionModelAbstract::multiplyByFx(Fx, A, out, op);
  }

  void default_multiplyFxTransposeBy(const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A,
                                     Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp op) const {
    return this->DifferentialActionModelAbstract::multiplyFxTransposeBy(Fx, A, out, op);
  }

  void default_multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp op) const {
    return this->DifferentialActionModelAbstract::multiplyByFu(Fu, A, out, op);
  }

  void default_multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                                     Eigen::Ref<MatrixXsRowMajor> out, const AssignmentOp op) const {
    return this->DifferentialActionModelAbstract::multiplyFuTransposeBy(Fu, A, out, op);
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

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(DifferentialActionModel_quasiStatic_wraps,
                                       DifferentialActionModelAbstract::quasiStatic_x, 2, 4)

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
