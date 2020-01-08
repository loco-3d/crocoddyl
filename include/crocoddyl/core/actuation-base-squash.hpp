#include <Eigen/Dense>

#include "crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {

class ActuationModelSquashingAbstract : public ActuationModelAbstract {
    public:
        ActuationModelSquashingAbstract(boost::shared_ptr<StateAbstract> state, const std::size_t& nu);
        virtual ~ActuationModelSquashingAbstract();

        virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                    const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
        virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                        const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                        const bool& recalc = true) = 0;

        virtual void calcSquash(const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
        virtual void calcSquashDiff(const Eigen::Ref<const Eigen::VectorXd>& u) = 0;
    
    protected:
        Eigen::VectorXd v_; // Squashing function output
        Eigen::MatrixXd dv_du_; // Squashing function Jacobian wrt control input u

};
}