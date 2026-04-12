#pragma once
#include "../core/activation_function.hpp"
#include "layer.hpp"
#include <utility>

namespace nn {

class ActivationLayer : public Layer {
private:
  AnyActivation activation_; // Сама функция (пока что это ReLU)
  Matrix input_cache_;       // Кэш для обратного прохода

public:
  explicit ActivationLayer(AnyActivation activation)
      : activation_(std::move(activation)) {}

  Matrix forward(const Matrix &input) override {
    input_cache_ = input;
    return activation_->forward(input);
  }

  Matrix predict(const Matrix &input) override {
    // Нам не нужен кэш, т.к. слой не линейный
    return activation_->forward(input);
  }

  Matrix backward(const Matrix &grad_output) override {
    // Передаем закэшированный X и пришедшую ошибку
    return activation_->backward(input_cache_, grad_output);
  }

  void update(Optimizer & /*opt*/) override {
    // Мы не умеем обновлять нелинейный слой(тут собственно нет весов)
  }
};

} // namespace nn
