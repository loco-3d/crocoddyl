///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

class CostModelContactFrictionCone : public CostModelAbstract {
 public:
  CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const FrictionCone& cone,
                               const FrameIndex& frame, const std::size_t& nu);
  CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const FrictionCone& cone,
                               const FrameIndex& frame);
  CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state, const FrictionCone& cone,
                               const FrameIndex& frame, const std::size_t& nu);
  CostModelContactFrictionCone(boost::shared_ptr<StateMultibody> state, const FrictionCone& cone,
                               const FrameIndex& frame);
  ~CostModelContactFrictionCone();

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u);
  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u);
  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  const FrictionCone& get_friction_cone() const;
  const FrameIndex& get_frame() const;
  void set_friction_cone(const FrictionCone& cone);
  void set_frame(const FrameIndex& frame);

 protected:
  FrictionCone friction_cone_;
  FrameIndex frame_;
};

struct CostDataContactFrictionCone : public CostDataAbstract {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  template <typename Model>
  CostDataContactFrictionCone(Model* const model, DataCollectorAbstract* const data)
      : CostDataAbstract(model, data),
        Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()),
        more_than_3_constraints(false) {
    Arr_Ru.fill(0);

    // Check that proper shared data has been passed
    DataCollectorContact* d = dynamic_cast<DataCollectorContact*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }

    // Avoids data casting at runtime
    std::string frame_name = model->get_state()->get_pinocchio().frames[model->get_frame()].name;
    bool found_contact = false;
    for (ContactModelMultiple::ContactDataContainer::iterator it = d->contacts->contacts.begin();
         it != d->contacts->contacts.end(); ++it) {
      if (it->second->frame == model->get_frame()) {
        ContactData3D* d3d = dynamic_cast<ContactData3D*>(it->second.get());
        if (d3d != NULL) {
          found_contact = true;
          contact = it->second;
          break;
        }
        ContactData6D* d6d = dynamic_cast<ContactData6D*>(it->second.get());
        if (d6d != NULL) {
          more_than_3_constraints = true;
          found_contact = true;
          contact = it->second;
          break;
        }
        throw_pretty("Domain error: there isn't defined at least a 3d contact for " + frame_name);
        break;
      }
    }
    if (!found_contact) {
      throw_pretty("Domain error: there isn't defined contact data for " + frame_name);
    }
  }

  boost::shared_ptr<ContactDataAbstract> contact;
  Eigen::MatrixXd Arr_Ru;
  bool more_than_3_constraints;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_FRICTION_CONE_HPP_
