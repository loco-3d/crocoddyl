#ifndef CROCODDYL_CORE_SQUASHING_BASE_HPP_
#define CROCODDYL_CORE_SQUASHING_BASE_HPP_

#include <stdexcept>
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

  struct SquashingDataAbstract; // forward declaration

  class SquashingModelAbstract {
   public:
    SquashingModelAbstract(boost::shared_ptr<ActuationModelAbstract> actuation, const std::size_t& ns);
    virtual ~SquashingModelAbstract();

    virtual void calc(const boost::shared_ptr<SquashingDataAbstract>& data,
              const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
    virtual void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data,
                  const Eigen::Ref<const Eigen::VectorXd>& u,
                  const bool& recalc = true) = 0;
    virtual boost::shared_ptr<SquashingDataAbstract> createData();

    const std::size_t& get_ns() const;
    const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  protected:
    std::size_t ns_;
    boost::shared_ptr<ActuationModelAbstract> actuation_; 
    Eigen::VectorXd out_ub_; // Squashing function upper bound
    Eigen::VectorXd out_lb_; // Squashing function lower bound
    Eigen::VectorXd in_ub_;  // Bound for the u variable (to apply using the Quadratic barrier)
    Eigen::VectorXd in_lb_;  // Bound for the u variable (to apply using the Quadratic  barrier)
    
  };

  struct SquashingDataAbstract {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    template <typename Model>
    explicit SquashingDataAbstract(Model* const model)
        : s(model->get_actuation()->get_nu()),
          ds_du(model->get_actuation()->get_nu(), model->get_actuation()->get_nu()) {
            s.fill(0);
            ds_du.fill(0);
    }
    virtual ~SquashingDataAbstract() {}

    Eigen::VectorXd s;
    Eigen::MatrixXd ds_du;
  };
}

#endif // CROCODDYL_CORE_SQUASHING_BASE_HPP_