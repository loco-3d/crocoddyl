
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

class CostModelFrameTranslation : public CostModelAbstract {
 public:
  CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state,
                            boost::shared_ptr<ActivationModelAbstract> activation, const FrameTranslation& xref,
                            const std::size_t& nu);
  CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state,
                            boost::shared_ptr<ActivationModelAbstract> activation, const FrameTranslation& xref);
  CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                            const std::size_t& nu);
  CostModelFrameTranslation(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref);
  ~CostModelFrameTranslation();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FrameTranslation& get_xref() const;
  void set_xref(const FrameTranslation& xref_in);

 private:
  FrameTranslation xref_;
};

struct CostDataFrameTranslation : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataFrameTranslation(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data), J(3, model->get_state()->get_nv()), fJf(6, model->get_state()->get_nv()) {
    J.fill(0);
    fJf.fill(0);
    // Check that proper shared data has been passed
    DataCollectorMultibody* d = dynamic_cast<DataCollectorMultibody*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::Data* pinocchio;
  pinocchio::Data::Matrix3x J;
  pinocchio::Data::Matrix6x fJf;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
