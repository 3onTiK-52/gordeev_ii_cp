#pragma once
#include "loss.hpp"

namespace nn {

class MSE : public Loss {
public:
  float forward(const Matrix &predictions, const Matrix &targets) override {
    Matrix diff = predictions - targets;
    // Переводим матрицу в режим array(), чтобы возвести каждый элемент в
    // квадрат, а затем просто берем среднее значение .mean()
    return diff.array().square().mean();
  }

  Matrix backward(const Matrix &predictions, const Matrix &targets) override {
    // N - общее количество элементов (размер батча * количество нейронов на
    // выходе)
    float N = predictions.rows() * predictions.cols();

    // Производная: 2 * (Y_pred - Y_true) / N
    return 2.0f * (predictions - targets) / N;
  }
};

} // namespace nn