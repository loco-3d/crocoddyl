///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COST_BASE_HPP_
#define CROCODDYL_MULTIBODY_COST_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"

namespace crocoddyl {

template <typename _Scalar>
class CostModelAbstractTpl {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                       const std::size_t& nu, const bool& with_residuals = true);
  CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActivationModelAbstract> activation,
                       const bool& with_residuals = true);
  CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nr, const std::size_t& nu,
                       const bool& with_residuals = true);
  CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nr,
                       const bool& with_residuals = true);
  ~CostModelAbstractTpl();

  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const boost::shared_ptr<ActivationModelAbstract>& get_activation() const;
  const std::size_t& get_nu() const;

 protected:
  boost::shared_ptr<StateMultibody> state_;
  boost::shared_ptr<ActivationModelAbstract> activation_;
  std::size_t nu_;
  bool with_residuals_;
  VectorXs unone_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<CostDataAbstract>& data, const VectorXs& x, const VectorXs& u = VectorXs()) {
    if (u.size() == 0) {
      calc(data, x);
    } else {
      calc(data, x, u);
    }
  }

  void calcDiff_wrap(const boost::shared_ptr<CostDataAbstract>& data, const VectorXs& x, const VectorXs& u) {
    calcDiff(data, x, u);
  }
  void calcDiff_wrap(const boost::shared_ptr<CostDataAbstract>& data, const VectorXs& x) { calcDiff(data, x, unone_); }

#endif
};

template <typename _Scalar>
struct CostDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataAbstractTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : shared(data),
        activation(model->get_activation()->createData()),
        cost(0.),
        Lx(model->get_state()->get_ndx()),
        Lu(model->get_nu()),
        Lxx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(model->get_state()->get_ndx(), model->get_nu()),
        Luu(model->get_nu(), model->get_nu()),
        r(model->get_activation()->get_nr()),
        Rx(model->get_activation()->get_nr(), model->get_state()->get_ndx()),
        Ru(model->get_activation()->get_nr(), model->get_nu()) {
    Lx.fill(0);
    Lu.fill(0);
    Lxx.fill(0);
    Lxu.fill(0);
    Luu.fill(0);
    r.fill(0);
    Rx.fill(0);
    Ru.fill(0);
  }
  virtual ~CostDataAbstractTpl() {}

  DataCollectorAbstract* shared;
  boost::shared_ptr<ActivationDataAbstract> activation;
  Scalar cost;
  VectorXs Lx;
  VectorXs Lu;
  MatrixXs Lxx;
  MatrixXs Lxu;
  MatrixXs Luu;
  VectorXs r;
  MatrixXs Rx;
  MatrixXs Ru;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/cost-base.hxx"

#endif  // CROCODDYL_MULTIBODY_COST_BASE_HPP_
