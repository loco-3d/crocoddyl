#ifndef CROCODDYL_CORE_DATA_SQUASHING_HPP_
#define CROCODDYL_CORE_DATA_SQUASHING_HPP_

#include <boost/shared_ptr.hpp>
#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/squashing-base.hpp"

namespace crocoddyl {

struct DataCollectorSquashing : virtual DataCollectorAbstract {
  DataCollectorSquashing(boost::shared_ptr<SquashingDataAbstract> squashing)
      : DataCollectorAbstract(), squashing(squashing) {}
  virtual ~DataCollectorSquashing() {}

  boost::shared_ptr<SquashingDataAbstract> squashing;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_DATA_SQUASHING_HPP_