#pragma once
#include "../core/types.hpp"

namespace nn {

class Optimizer {
public:
  virtual ~Optimizer() = default;

  virtual void update(Matrix &params, const Matrix &grads) = 0;
};

} // namespace nn