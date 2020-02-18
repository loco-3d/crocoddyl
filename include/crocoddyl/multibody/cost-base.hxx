///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template<typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state,
                                     boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& nu,
                                     const bool& with_residuals)
    : state_(state),
      activation_(activation),
      nu_(nu),
      with_residuals_(with_residuals),
      unone_(VectorXs::Zero(nu)) {}

template<typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state,
                                     boost::shared_ptr<ActivationModelAbstract> activation, const bool& with_residuals)
    : state_(state),
      activation_(activation),
      nu_(state->get_nv()),
      with_residuals_(with_residuals),
      unone_(VectorXs::Zero(state->get_nv())) {}
  
template<typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nr,
                                     const std::size_t& nu, const bool& with_residuals)
    : state_(state),
      activation_(boost::make_shared<ActivationModelQuad>(nr)),
      nu_(nu),
      with_residuals_(with_residuals),
      unone_(VectorXs::Zero(nu)) {}
  
template<typename Scalar>
CostModelAbstractTpl<Scalar>::CostModelAbstractTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nr,
                                     const bool& with_residuals)
    : state_(state),
      activation_(boost::make_shared<ActivationModelQuad>(nr)),
      nu_(state->get_nv()),
      with_residuals_(with_residuals),
      unone_(VectorXs::Zero(state->get_nv())) {}
  
template<typename Scalar>
CostModelAbstractTpl<Scalar>::~CostModelAbstractTpl() {}

template<typename Scalar>  
void CostModelAbstractTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                        const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template<typename Scalar>  
void CostModelAbstractTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}
  
template<typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelAbstractTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return boost::make_shared<CostDataAbstract>(this, data);
}

template<typename Scalar>  
const boost::shared_ptr<StateMultibodyTpl<Scalar> >& CostModelAbstractTpl<Scalar>::get_state() const { return state_; }

template<typename Scalar>  
const boost::shared_ptr<ActivationModelAbstractTpl<Scalar> >& CostModelAbstractTpl<Scalar>::get_activation() const { return activation_; }

template<typename Scalar>  
const std::size_t& CostModelAbstractTpl<Scalar>::get_nu() const { return nu_; }

}  // namespace crocoddyl
