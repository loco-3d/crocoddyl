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
  CostDataFramePlacement(Model* const model, pinocchio::Data* const data) : CostDataAbstract(model, data) {}

  pinocchio::Motion::Vector6 r;
  pinocchio::SE3 rMf;
  pinocchio::SE3::Matrix6 J;
  pinocchio::SE3::Matrix6 oJf;
  pinocchio::SE3::Matrix6 rJf;
  pinocchio::SE3::Matrix6 fJf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
