#pragma once
#include "loss.hpp"
#include <cmath>

namespace nn {

class CrossEntropy : public Loss {
private:
  // Превращаем ответы сети в вероятности с суммой 1
  Matrix compute_softmax(const Matrix &logits) {
    Matrix probs = logits;
    for (int i = 0; i < probs.rows(); ++i) {
      // Вычитаем максимум, чтобы избежать переполнения памяти
      // при возведении в экспоненту (e^1000 выдаст NaN)
      float max_val = probs.row(i).maxCoeff();
      probs.row(i) = (probs.row(i).array() - max_val).exp();

      float sum = probs.row(i).sum();
      probs.row(i) /= sum;
    }
    return probs;
  }

public:
  float forward(const Matrix &predictions, const Matrix &targets) override {
    Matrix probs = compute_softmax(predictions);

    float loss = 0.0f;
    int N = predictions.rows();

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < predictions.cols(); ++j) {
        if (targets(i, j) > 0.0f) {
          // Добавляем 1e-7, чтобы случайно не получить логарифм от нуля
          loss -= targets(i, j) * std::log(probs(i, j) + 1e-7f);
        }
      }
    }
    return loss / N;
  }

  Matrix backward(const Matrix &predictions, const Matrix &targets) override {
    Matrix probs = compute_softmax(predictions);
    int N = predictions.rows();
    return (probs - targets) / N;
  }
};

} // namespace nn
