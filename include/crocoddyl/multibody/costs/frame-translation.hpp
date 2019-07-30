
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
// #include <pinocchio/spatial/se3.hpp>

namespace crocoddyl {

struct FrameTranslation {
  FrameTranslation(const unsigned int& frame, const Eigen::Vector3d& oxf) : frame(frame), oxf(oxf) {}
  unsigned int frame;
  Eigen::Vector3d oxf;
};

class CostModelFrameTranslation : public CostModelAbstract {
 public:
  CostModelFrameTranslation(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                            const FrameTranslation& xref, const unsigned int& nu);
  CostModelFrameTranslation(pinocchio::Model* const model, ActivationModelAbstract* const activation,
                            const FrameTranslation& xref);
  CostModelFrameTranslation(pinocchio::Model* const model, const FrameTranslation& xref, const unsigned int& nu);
  CostModelFrameTranslation(pinocchio::Model* const model, const FrameTranslation& xref);
  ~CostModelFrameTranslation();

  void calc(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const FrameTranslation& get_xref() const;

 private:
  FrameTranslation xref_;
};

struct CostDataFrameTranslation : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFrameTranslation(Model* const model, pinocchio::Data* const data)
      : CostDataAbstract(model, data), J(6, model->get_nv()), fJf(6, model->get_nv()) {
    J.fill(0);
    fJf.fill(0);
  }

  pinocchio::Data::Matrix3x J;
  pinocchio::Data::Matrix6x fJf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_