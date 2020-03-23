#ifndef CROCODDYL_CORE_SQUASHING_BASE_HPP_
#define CROCODDYL_CORE_SQUASHING_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

// struct SquashingDataAbstractTpl; // forward declaration

template <typename _Scalar>
class SquashingModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef SquashingDataAbstractTpl<Scalar> SquashingDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  SquashingModelAbstractTpl(const std::size_t& ns): ns_(ns) {
    if (ns_ == 0) {
      throw_pretty("Invalid argument: "
                << "ns cannot be zero");
    }
  };
  virtual ~SquashingModelAbstractTpl(){};

  virtual void calc(const boost::shared_ptr<SquashingDataAbstract>& data,
              const Eigen::Ref<const VectorXs>& u) = 0;
  virtual void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data,
                  const Eigen::Ref<const VectorXs>& u,
                  const bool& recalc = true) = 0;
  virtual boost::shared_ptr<SquashingDataAbstract> createData() {
    return boost::make_shared<SquashingDataAbstract>(this);
  }

  const std::size_t& get_ns() const;
  const VectorXs& get_in_lb() const { return in_lb_;};
  const VectorXs& get_in_ub() const { return in_ub_;};

  void set_in_lb(const VectorXs& in_lb) { in_lb_ = in_lb; };
  void set_in_ub(const VectorXs& in_ub) { in_ub_ = in_ub; };

 protected:
  std::size_t ns_;
  VectorXs out_ub_; // Squashing function upper bound
  VectorXs out_lb_; // Squashing function lower bound
  VectorXs in_ub_;  // Bound for the u variable (to apply using the Quadratic barrier)
  VectorXs in_lb_;  // Bound for the u variable (to apply using the Quadratic barrier)

#ifdef PYTHON_BINDINGS

 public:
  void calc_wrap(const boost::shared_ptr<SquashingDataAbstract>& data, 
  const VectorXs& u) {
    calc(data, u);
  }

  void calcDiff_wrap(const boost::shared_ptr<SquashingDataAbstract>& data,
                     const Eigen::VectorXs& u, const bool& recalc = true) {
    calcDiff(data, u, recalc);
  }

#endif
};

template <typename _Scalar>
struct SquashingDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  
  template <template <typename Scalar> class Model>
  explicit SquashingDataAbstractTpl(Model<Scalar>* const model)
      : s(model->get_ns()),
        ds_du(model->get_ns(), model->get_ns()) {
    s.setZero();
    ds_du.setZero();
  }
  virtual ~SquashingDataAbstractTpl() {}

  VectorXs s;
  MatrixXs ds_du;
};

} // namespace crocoddyl

#endif // CROCODDYL_CORE_SQUASHING_BASE_HPP_