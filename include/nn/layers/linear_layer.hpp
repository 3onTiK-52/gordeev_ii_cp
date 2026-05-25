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

  void save(std::ostream &os) const override {
    // 1. Сохраняем матрицу весов
    Matrix::Index w_rows = weights_.rows();
    Matrix::Index w_cols = weights_.cols();
    // Записываем размеры (метаданные)
    os.write(reinterpret_cast<const char *>(&w_rows), sizeof(w_rows));
    os.write(reinterpret_cast<const char *>(&w_cols), sizeof(w_cols));
    // Записываем саму матрицу весов побайтово
    os.write(reinterpret_cast<const char *>(weights_.data()),
             w_rows * w_cols * sizeof(float));

    // 2. Сохраняем вектор сдвигов (bias)
    Matrix::Index b_rows = biases_.rows();
    Matrix::Index b_cols = biases_.cols();
    // Записываем размеры
    os.write(reinterpret_cast<const char *>(&b_rows), sizeof(b_rows));
    os.write(reinterpret_cast<const char *>(&b_cols), sizeof(b_cols));
    // Записываем данные сдвигов
    os.write(reinterpret_cast<const char *>(biases_.data()),
             b_rows * b_cols * sizeof(float));
  }

  void load(std::istream &is) override {
    // 1. Читаем матрицу весов
    Matrix::Index w_rows, w_cols;
    is.read(reinterpret_cast<char *>(&w_rows), sizeof(w_rows));
    is.read(reinterpret_cast<char *>(&w_cols), sizeof(w_cols));

    // Проверяем, совпадают ли размеры из файла с текущей архитектурой слоя
    if (w_rows != weights_.rows() || w_cols != weights_.cols()) {
      throw std::runtime_error(
          "Ошибка загрузки весов: несоответствие размеров матрицы весов.");
    }
    is.read(reinterpret_cast<char *>(weights_.data()),
            w_rows * w_cols * sizeof(float));

    // 2. Читаем вектор сдвигов
    Matrix::Index b_rows, b_cols;
    is.read(reinterpret_cast<char *>(&b_rows), sizeof(b_rows));
    is.read(reinterpret_cast<char *>(&b_cols), sizeof(b_cols));

    if (b_rows != biases_.rows() || b_cols != biases_.cols()) {
      throw std::runtime_error(
          "Ошибка загрузки весов: несоответствие размеров вектора сдвигов.");
    }
    is.read(reinterpret_cast<char *>(biases_.data()),
            b_rows * b_cols * sizeof(float));
  }
};

} // namespace nn
