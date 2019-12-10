///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/numdiff/cost.hpp"

namespace crocoddyl {

CostNumDiffModel::CostNumDiffModel(const boost::shared_ptr<CostModelAbstract>& model):
  CostModelAbstract(model->get_state(), model->get_activation(), model->get_nu(), model->get_with_residuals()),
  model_(model)
{
  disturbance_ = std::sqrt(2.0 * std::numeric_limits<double>::epsilon());  
}

CostNumDiffModel::~CostNumDiffModel(){}

void CostNumDiffModel::calc(const boost::shared_ptr<CostNumDiffData>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u)
{

}

void CostNumDiffModel::calcDiff(const boost::shared_ptr<CostNumDiffData>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                        const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc)
{

}

boost::shared_ptr<CostDataAbstract> CostNumDiffModel::createData(DataCollectorAbstract* const data)
{

}

const boost::shared_ptr<CostModelAbstract>& CostNumDiffModel::get_model() const { return model_; }

const double& CostNumDiffModel::get_disturbance() const { return disturbance_; }

void CostNumDiffModel::set_disturbance(const double& disturbance) { disturbance_ = disturbance; }

bool CostNumDiffModel::get_with_gauss_approx() { return activation_->get_nr() > 0; }

void CostNumDiffModel::set_reevals(const std::vector<ReevaluationFunction>& reevals){ reevals_ = reevals; }

}
