#include <pinocchio/multibody/geometry.hpp>

namespace colmpc {

struct GeometryDataWrapper {
  pinocchio::GeometryData* geometry = nullptr;

  void setGeometry(pinocchio::GeometryData* g) {
    if (geometry != nullptr && g != geometry) {
      // This is not strictly required. However, setting the geom data twice
      // might indicate a bug.
      throw std::invalid_argument("Cannot set twice geometry data.");
    }
    geometry = g;
  }
};
}  // namespace colmpc
