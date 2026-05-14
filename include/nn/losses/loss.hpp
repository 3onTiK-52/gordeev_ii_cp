#pragma once
#include "../core/types.hpp"

namespace nn {

class Loss {
public:
  virtual ~Loss() = default;

  // Считает ошибку. Возвращает одно число (float)
  virtual float forward(const Matrix &predictions, const Matrix &targets) = 0;

  // Считает градиент. Возвращает матрицу dY, которая пойдет в слои сети
  virtual Matrix backward(const Matrix &predictions, const Matrix &targets) = 0;
};

} // namespace nn