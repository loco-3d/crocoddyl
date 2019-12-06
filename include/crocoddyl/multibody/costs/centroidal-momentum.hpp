///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"

namespace crocoddyl {

typedef Eigen::Matrix<double, 6, 1> Vector6d;

class CostModelCentroidalMomentum : public CostModelAbstract {
 public:
  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state,
                              boost::shared_ptr<ActivationModelAbstract> activation, const Vector6d& mref,
                              const std::size_t& nu);
  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state,
                              boost::shared_ptr<ActivationModelAbstract> activation, const Vector6d& mref);
  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state, const Vector6d& mref, const std::size_t& nu);
  CostModelCentroidalMomentum(boost::shared_ptr<StateMultibody> state, const Vector6d& mref);
  ~CostModelCentroidalMomentum();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const Vector6d& get_href() const;
  void set_href(const Vector6d& mref_in);

 private:
  Vector6d href_;
};

struct CostDataCentroidalMomentum : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataCentroidalMomentum(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data),
        dhd_dq(6, model->get_state()->get_nv()),
        dhd_dv(6, model->get_state()->get_nv()) {
    dhd_dq.fill(0);
    dhd_dv.fill(0);

    // Check that proper shared data has been passed
    DataCollectorMultibody* d = dynamic_cast<DataCollectorMultibody*>(shared);
    if (d == NULL) {
      throw std::invalid_argument("the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::Data* pinocchio;
  pinocchio::Data::Matrix6x dhd_dq;
  pinocchio::Data::Matrix6x dhd_dv;
  pinocchio::Data::Matrix6x Arr_Rx;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
