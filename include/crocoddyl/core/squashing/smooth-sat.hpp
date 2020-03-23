#ifndef CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_
#define CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_

#include "crocoddyl/core/squashing-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class SquashingModelSmoothSatTpl : public SquashingModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef SquashingModelAbstractTpl<Scalar> Base;
  typedef SquashingDataAbstractTpl<Scalar> SquashingDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  explicit SquashingModelSmoothSatTpl(const Eigen::Ref<const VectorXs>& out_lb, 
                          const Eigen::Ref<const VectorXs>& out_ub,
                          const std::size_t& ns) : Base(ns)
  {
    out_lb_ = out_lb;
    out_ub_ = out_ub;

    in_lb_ = out_lb_;
    in_ub_ = out_ub_;
    
    smooth_ = 0.1;

    d_ = (out_ub_ - out_lb_)*smooth_;
    a_ = d_.array() * d_.array();

  }
  
  ~SquashingModelSmoothSatTpl();

  void calc(const boost::shared_ptr<SquashingDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& u)
  {
    // Squashing function used: "Smooth abs": 
    // s(u) = 0.5*(lb + ub + sqrt(smooth + (u - lb)^2) - sqrt(smooth + (u - ub)^2))
      data->s = 0.5*(Eigen::sqrt(Eigen::pow((u - out_lb_).array(), 2) + a_.array()) - Eigen::sqrt(Eigen::pow((u - out_ub_).array(), 2) + a_.array()) + out_lb_.array() + out_ub_.array()); 
  }
  
  void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data,
                  const Eigen::Ref<const VectorXs>& u,
                  const bool& recalc = true)
  {
    if (recalc)
    {
      calc(data, u);
    }
    
    data->ds_du.diagonal()= 0.5*(Eigen::pow(a_.array() + Eigen::pow((u - out_lb_).array(), 2), -0.5).array() * (u - out_lb_).array() - Eigen::pow(a_.array() + Eigen::pow((u - out_ub_).array(), 2), -0.5).array() * (u - out_ub_).array());
  }
  
  const Scalar& get_smooth() const {return smooth_;};
  void set_smooth(const Scalar& smooth)
  {
    if (smooth < 0.) {
      throw_pretty("Invalid argument: " << "Smooth value has to be positive");
    }
    smooth_ = smooth;
    
    d_ = (out_ub_ - out_lb_)*smooth_;
    a_ = d_.array() * d_.array();

    in_lb_ = out_lb_;
    in_ub_ = out_ub_;
  }

  const VectorXs& get_d() const {return d_;};

 private:
  VectorXs a_;
  VectorXs d_;

  Scalar smooth_;
 protected:
  using Base::out_ub_;
  using Base::out_lb_;

  using Base::in_ub_;
  using Base::in_lb_;

};
}

#endif // CROCODDYL_CORE_SQUASHING_SMOOTH_SAT_HPP_