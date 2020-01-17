///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_IMPULSE_COST_BASE_HPP_
#define CROCODDYL_MULTIBODY_IMPULSE_COST_BASE_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

struct ImpulseCostDataAbstract;  // forward declaration

class ImpulseCostModelAbstract {
 public:
  ImpulseCostModelAbstract(boost::shared_ptr<StateMultibody> state,
                           boost::shared_ptr<ActivationModelAbstract> activation, const bool& with_residuals = true);
  ImpulseCostModelAbstract(boost::shared_ptr<StateMultibody> state, const std::size_t& nr,
                           const bool& with_residuals = true);
  ~ImpulseCostModelAbstract();

  virtual void calc(const boost::shared_ptr<ImpulseCostDataAbstract>& data,
                    const Eigen::Ref<const Eigen::VectorXd>& x) = 0;
  virtual void calcDiff(const boost::shared_ptr<ImpulseCostDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& x, const bool& recalc = true) = 0;
  virtual boost::shared_ptr<ImpulseCostDataAbstract> createData(DataCollectorAbstract* const data);

  const boost::shared_ptr<StateMultibody>& get_state() const;
  const boost::shared_ptr<ActivationModelAbstract>& get_activation() const;

 protected:
  boost::shared_ptr<StateMultibody> state_;
  boost::shared_ptr<ActivationModelAbstract> activation_;
  bool with_residuals_;

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<ImpulseCostDataAbstract>& data, const Eigen::VectorXd& x) { calc(data, x); }

  void calcDiff_wrap(const boost::shared_ptr<ImpulseCostDataAbstract>& data, const Eigen::VectorXd& x,
                     const bool& recalc) {
    calcDiff(data, x, recalc);
  }
  void calcDiff_wrap(const boost::shared_ptr<ImpulseCostDataAbstract>& data, const Eigen::VectorXd& x) {
    calcDiff(data, x, true);
  }

#endif
};

struct ImpulseCostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  ImpulseCostDataAbstract(Model* const model, DataCollectorAbstract* const data)
      : shared(data),
        activation(model->get_activation()->createData()),
        cost(0.),
        Lx(model->get_state()->get_ndx()),
        Lu(0),
        Lxx(model->get_state()->get_ndx(), model->get_state()->get_ndx()),
        Lxu(model->get_state()->get_ndx(), 0),
        Luu(0, 0),
        r(model->get_activation()->get_nr()),
        Rx(model->get_activation()->get_nr(), model->get_state()->get_ndx()),
        Ru(model->get_activation()->get_nr(), 0) {
    Lx.fill(0);
    Lu.fill(0);
    Lxx.fill(0);
    Lxu.fill(0);
    Luu.fill(0);
    r.fill(0);
    Rx.fill(0);
    Ru.fill(0);

    // Check that proper shared data has been passed
    DataCollectorImpulse* d = dynamic_cast<DataCollectorImpulse*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorImpulse");
    } else {
      impulses = d->impulses;
    }
  }
  virtual ~ImpulseCostDataAbstract() {}

  DataCollectorAbstract* shared;
  boost::shared_ptr<ImpulseDataMultiple> impulses;
  boost::shared_ptr<ActivationDataAbstract> activation;
  double cost;
  Eigen::VectorXd Lx;
  Eigen::VectorXd Lu;
  Eigen::MatrixXd Lxx;
  Eigen::MatrixXd Lxu;
  Eigen::MatrixXd Luu;
  Eigen::VectorXd r;
  Eigen::MatrixXd Rx;
  Eigen::MatrixXd Ru;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COST_BASE_HPP_
