#pragma once
#include "optimizer.hpp"

namespace nn {

class SGD : public Optimizer {
private:
  float lr_;           // Скорость обучения - Learning Rate
  float weight_decay_; // Коэффициент L2-регуляризации

public:
  explicit SGD(float lr = 0.01f, float weight_decay = 0.0f)
      : lr_(lr), weight_decay_(weight_decay) {}

  void update(Matrix &params, const Matrix &grads) override {
    // Формула шага обучения: W_n = W_n-1 - lr * (Градиент + L2-рег.)
    if (weight_decay_ > 0.0f) {
      params -= lr_ * (grads + weight_decay_ * params);
    } else {
      params -= lr_ * grads;
    }
  }
};

} // namespace nn