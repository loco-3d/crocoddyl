///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_
#define CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace unittest {

struct PinocchioModelTypes {
  enum Type {
    Hector,
    TalosArm,
    HyQ,
    Talos,
    RandomHumanoid,
    NbPinocchioModelTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbPinocchioModelTypes);
    for (int i = 0; i < NbPinocchioModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, PinocchioModelTypes::Type type);

class PinocchioModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PinocchioModelFactory(PinocchioModelTypes::Type type);
  ~PinocchioModelFactory();

  void construct_model(const std::string& urdf_file = "",
                       const std::string& srdf_file = "",
                       bool free_flyer = true);

  std::shared_ptr<pinocchio::Model> create() const;
  std::vector<std::string> get_frame_names() const;
  std::vector<std::size_t> get_frame_ids() const;
  std::size_t get_contact_nc() const;

 private:
  std::shared_ptr<pinocchio::Model>
      model_;  //!< The pointer to the state in testing
  std::vector<std::string> frame_name_;  //!< Frame name for unittesting
  std::vector<std::size_t> frame_id_;    //!< Frame id for unittesting
  std::size_t contact_nc_;               //!< Dimension of the contact
};

/**
 * @brief Compute all the pinocchio data needed for the numerical
 * differentiation. We use the address of the object to avoid a copy from the
 * "boost::bind".
 *
 * @param model[in]  Pinocchio model
 * @param data[out]  Pinocchio data
 * @param x[in]      State vector
 * @param u[in]      Control vector
 */
template <typename Scalar, int Options,
          template <typename, int> class JointCollectionTpl>
void updateAllPinocchio(
    pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl>* const model,
    pinocchio::DataTpl<Scalar, Options, JointCollectionTpl>* data,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& u =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>());

}  // namespace unittest
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "pinocchio_model.hxx"

#endif  // CROCODDYL_PINOCCHIO_MODEL_FACTORY_HPP_
