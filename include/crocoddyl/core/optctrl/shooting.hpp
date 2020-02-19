///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
#define CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
#include "crocoddyl/core/fwd.hpp"
#include <stdexcept>
#include <vector>
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ShootingProblemTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  
  ShootingProblemTpl(const VectorXs& x0,
                     const std::vector<boost::shared_ptr<ActionModelAbstract> >& running_models,
                     boost::shared_ptr<ActionModelAbstract> terminal_model);
  ~ShootingProblemTpl();

  Scalar calc(const std::vector<VectorXs>& xs,
              const std::vector<VectorXs>& us);
  Scalar calcDiff(const std::vector<VectorXs>& xs,
                  const std::vector<VectorXs>& us);
  void rollout(const std::vector<VectorXs>& us,
               std::vector<VectorXs>& xs);
  std::vector<VectorXs> rollout_us(const std::vector<VectorXs>& us);

  const std::size_t& get_T() const;
  const VectorXs& get_x0() const;
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& get_runningModels() const;
  const boost::shared_ptr<ActionModelAbstract>& get_terminalModel() const;
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& get_runningDatas() const;
  const boost::shared_ptr<ActionDataAbstract>& get_terminalData() const;

  void set_x0(const VectorXs& x0_in);
  void set_runningModels(const std::vector<boost::shared_ptr<ActionModelAbstract> >& models);
  void set_terminalModel(boost::shared_ptr<ActionModelAbstract> model);

 protected:
  Scalar cost_;
  std::size_t T_;
  VectorXs x0_;
  boost::shared_ptr<ActionModelAbstract> terminal_model_;
  boost::shared_ptr<ActionDataAbstract> terminal_data_;
  std::vector<boost::shared_ptr<ActionModelAbstract> > running_models_;
  std::vector<boost::shared_ptr<ActionDataAbstract> > running_datas_;

 private:
  void allocateData();
};

  
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/core/optctrl/shooting.hxx>

#endif  // CROCODDYL_CORE_OPTCTRL_SHOOTING_HPP_
