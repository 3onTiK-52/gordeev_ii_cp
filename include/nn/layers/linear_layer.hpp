#pragma once
#include "../optimizers/optimizer.hpp"
#include "layer.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace nn {

class LinearLayer : public Layer {
private:
  Matrix weights_;     // Матрица весов W
  Matrix biases_;      // Вектор сдвигов(W_0) b
  Matrix input_cache_; // Кэш X

  Matrix grad_weights_;
  Matrix grad_biases_;

public:
  LinearLayer(int input_size, int output_size) {
    weights_ = Matrix::Random(input_size, output_size) / std::sqrt(input_size);
    biases_ = Matrix::Zero(1, output_size);

    grad_weights_ = Matrix::Zero(input_size, output_size);
    grad_biases_ = Matrix::Zero(1, output_size);
  }

  Matrix forward(const Matrix &input) override {
    input_cache_ = input;
    return (input * weights_).rowwise() + biases_.row(0);
  }

  Matrix predict(const Matrix &input) override {
    return (input * weights_).rowwise() + biases_.row(0);
  }

  Matrix backward(const Matrix &grad_output) override {
    // 1. Считаем градиент по весам: dW = X^T * dY
    grad_weights_ = input_cache_.transpose() * grad_output;

    // 2. Считаем градиент по сдвигам: db = сумма dY
    grad_biases_ = grad_output.colwise().sum();

    // 3. Проталкиваем градиент для предыдущего слоя: dX = dY * W^T
    return grad_output * weights_.transpose();
  }

  void update(Optimizer &opt) override {
    opt.update(weights_, grad_weights_);
    opt.update(biases_, grad_biases_);
  }

  // Геттеры (для тестов, чтобы проверять правильность градиентов)
  const Matrix &get_weights() const { return weights_; }
  const Matrix &get_biases() const { return biases_; }
  const Matrix &get_grad_weights() const { return grad_weights_; }
  const Matrix &get_grad_biases() const { return grad_biases_; }

  // Сеттеры для ручного ввода весов (для тестов)
  void set_weights(const Matrix &w) { weights_ = w; }
  void set_biases(const Matrix &b) { biases_ = b; }
};

} // namespace nn
