///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include <stdexcept>
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

struct ActuationDataAbstract;  // forward declaration

class ActuationModelAbstract {
 public:
  ActuationModelAbstract(boost::shared_ptr<StateAbstract> state, const std::size_t& nu);
  virtual ~ActuationModelAbstract();

  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
  virtual boost::shared_ptr<ActuationDataAbstract> createData();

  const std::size_t& get_nu() const;
  const boost::shared_ptr<StateAbstract>& get_state() const;

 protected:
  std::size_t nu_;
  boost::shared_ptr<StateAbstract> state_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::VectorXd& x,
                 const Eigen::VectorXd& u) {
    calc(data, x, u);
  }

  void calcDiff_wrap(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u) {
    calcDiff(data, x, u);
  }

#endif
};

struct ActuationDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  explicit ActuationDataAbstract(Model* const model)
      : tau(model->get_state()->get_nv()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        dtau_du(model->get_state()->get_nv(), model->get_nu()) {
    tau.fill(0);
    dtau_dx.fill(0);
    dtau_du.fill(0);
  }
  virtual ~ActuationDataAbstract() {}

  Eigen::VectorXd tau;
  Eigen::MatrixXd dtau_dx;
  Eigen::MatrixXd dtau_du;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTUATION_BASE_HPP_
