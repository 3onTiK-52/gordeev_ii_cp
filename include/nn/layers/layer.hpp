#pragma once
#include <Eigen/Dense>

namespace nn {

class Optimizer;

using Matrix = Eigen::MatrixXf;

class Layer {
public:
  virtual ~Layer() = default;

  // Прямой проход: вычисляет выход слоя и сохраняет внутренний кэш для будущего
  // изменения весов
  virtual Matrix forward(const Matrix &input) = 0;

  // Предсказание: только вычисляет выход (быстрее, без сохранения кэша)
  virtual Matrix predict(const Matrix &input) = 0;

  // Обратный проход: принимает градиент от следующего слоя,
  // считает градиенты своих весов и возвращает градиент по входу
  virtual Matrix backward(const Matrix &grad_output) = 0;

  // Обновление весов слоя через переданный оптимизатор
  virtual void update(Optimizer &opt) = 0;
};

} // namespace nn