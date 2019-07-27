///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include <pinocchio/spatial/se3.hpp>

namespace crocoddyl {

struct FramePlacement {
  FramePlacement(const unsigned int& frame, const pinocchio::SE3& oMf) : frame(frame), oMf(oMf) {}
  unsigned int frame;
  pinocchio::SE3 oMf;
};

class CostModelFramePlacement : public CostModelAbstract {
 public:
  CostModelFramePlacement(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                          const FramePlacement& Fref, const unsigned int& nu);
  CostModelFramePlacement(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                          const FramePlacement& Fref);
  CostModelFramePlacement(pinocchio::Model* const model, const FramePlacement& Fref, const unsigned int& nu);
  CostModelFramePlacement(pinocchio::Model* const model, const FramePlacement& Fref);
  ~CostModelFramePlacement();

  void calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const FramePlacement& get_Mref() const;

 private:
  FramePlacement Mref_;
};

struct CostDataFramePlacement : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFramePlacement(Model* const model, pinocchio::Data* const data)
      : CostDataAbstract(model, data), J(6, model->get_nv()), rJf(6, 6), fJf(6, model->get_nv()) {
    r.fill(0);
    J.fill(0);
    rJf.fill(0);
    fJf.fill(0);
  }

  pinocchio::Motion::Vector6 r;
  pinocchio::SE3 rMf;
  pinocchio::Data::Matrix6x J;
  pinocchio::Data::Matrix6 rJf;
  pinocchio::Data::Matrix6x fJf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
