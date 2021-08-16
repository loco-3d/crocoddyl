///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_HPP_

#include <stdexcept>

#ifdef PINOCCHIO_WITH_CPPAD_SUPPORT  // TODO(cmastalli): Removed after merging Pinocchio v.2.4.8
#include <pinocchio/codegen/cppadcg.hpp>
#endif

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
namespace crocoddyl {

template <typename _Scalar>
class DifferentialActionModelFreeInvDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataFreeInvDynamicsTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ConstraintModelResidualTpl<Scalar> ConstraintModelResidual;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  DifferentialActionModelFreeInvDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                            boost::shared_ptr<ActuationModelAbstract> actuation,
                                            boost::shared_ptr<CostModelSum> costs,
                                            boost::shared_ptr<ConstraintModelManager> constraints = nullptr);
  virtual ~DifferentialActionModelFreeInvDynamicsTpl();

  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;
  const boost::shared_ptr<CostModelSum>& get_costs() const;
  const boost::shared_ptr<ConstraintModelManager>& get_constraints() const;
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  template <typename _Scalar> 
  class ResidualModelRneaTpl : public ResidualModelAbstractTpl<_Scalar>{
    public:
      typedef _Scalar Scalar;
      typedef MathBaseTpl<Scalar> MathBase;
      typedef ResidualModelAbstractTpl<Scalar> rneaBase;
      typedef ResidualDataRneaTpl<Scalar> rneaData;
      typedef StateMultibodyTpl<Scalar> StateMultibody;
      typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
      typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
      typedef typename MathBase::Vector3s Vector3s;
      typedef typename MathBase::VectorXs VectorXs;
    
    ResidualModelRneaTpl(boost::shared_ptr<StateMultibodyTpl> state, const std::size_t nu);
    virtual ~ResidualModelRneaTpl();

    virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                      const Eigen::Ref<const VectorXs>& u);

    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                          const Eigen::Ref<const VectorXs>& u);
    
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);
    
    virtual void print(std::ostream& os) const;
    
    protected:
      using Base::nu_;
      using Base::state_;
      using Base::u_dependent_;
      using Base::unone_;
      using Base::v_dependent_;
  }



  /**
   * @brief Print relevant information of the free forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::ng_;     //!< Number of inequality constraints
  using Base::nh_;     //!< Number of equality constraints
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  boost::shared_ptr<ActuationModelAbstract> actuation_;
  boost::shared_ptr<CostModelSum> costs_;
  boost::shared_ptr<ConstraintModelManager> constraints_;
  pinocchio::ModelTpl<Scalar>& pinocchio_;
  bool without_armature_;
  VectorXs armature_;
};

template <typename _Scalar>
struct DifferentialActionDataFreeInvDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataFreeInvDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_actuation()->createData()),
        costs(model->get_costs()->createData(&multibody)),
        Minv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        u_drift(model->get_nu()),
        dtau_dx(model->get_nu(), model->get_state()->get_ndx()),
        tmp_xstatic(model->get_state()->get_nx()).
        tmp_ustatic(model->get_nu())
         {
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    dtau_dx.setZero();
    tmp_xstatic.setZero();
    tmp_ustatic.setZero();
  }



  template <typename _Scalar>
  struct ResidualDataRneaTpl : public ResidualDataAbstractTpl<_Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> rneaBase;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::Matrix3xs Matrix3xs;
  
    template <template <typename Scalar> class Model>
    ResidualDataRneaTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : rneaBase(model, data) {
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
  };

  
  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorActMultibodyTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;
  boost::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;
  MatrixXs dtau_dx;
  VectorXs tmp_xstatic;
  VectorXs tmp_ustatic;
  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/free-invdyn.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_FREE_FWDDYN_HPP_
