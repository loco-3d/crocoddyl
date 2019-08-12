
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
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

class CostModelFrameTranslation : public CostModelAbstract {
 public:
  CostModelFrameTranslation(StateMultibody& state, ActivationModelAbstract& activation, const FrameTranslation& xref,
                            unsigned int const& nu);
  CostModelFrameTranslation(StateMultibody& state, ActivationModelAbstract& activation, const FrameTranslation& xref);
  CostModelFrameTranslation(StateMultibody& state, const FrameTranslation& xref, const unsigned int& nu);
  CostModelFrameTranslation(StateMultibody& state, const FrameTranslation& xref);
  ~CostModelFrameTranslation();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(pinocchio::Data* const data);

  const FrameTranslation& get_xref() const;

 private:
  FrameTranslation xref_;
  const unsigned int& nv_;
};

struct CostDataFrameTranslation : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFrameTranslation(Model* const model, pinocchio::Data* const data)
      : CostDataAbstract(model, data), J(3, model->get_state().get_nv()), fJf(6, model->get_state().get_nv()) {
    J.fill(0);
    fJf.fill(0);
  }

  pinocchio::Data::Matrix3x J;
  pinocchio::Data::Matrix6x fJf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
