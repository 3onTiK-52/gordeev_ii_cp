#pragma once
#include "../losses/loss.hpp"
#include "../optimizers/optimizer.hpp"
#include "../sequential.hpp"
#include <iomanip>
#include <iostream>

namespace nn {

class Trainer {
private:
  Sequential &model_;
  Loss &criterion_;
  Optimizer &optimizer_;

public:
  Trainer(Sequential &model, Loss &criterion, Optimizer &optimizer)
      : model_(model), criterion_(criterion), optimizer_(optimizer) {}

  void train(const Matrix &X, const Matrix &Y_true, int epochs,
             int report_every = 50) {
    int subset_size = X.rows();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
      // Шаг 1: Прямой проход
      Matrix Y_pred = model_.forward(X);

      // Шаг 2: Вычисление ошибки
      float loss = criterion_.forward(Y_pred, Y_true);

      // Шаг 3: Обратный проход (вычисление градиентов)
      Matrix grad = criterion_.backward(Y_pred, Y_true);
      model_.backward(grad);

      // Шаг 4: Обновление весов оптимизатором
      model_.update(optimizer_);

      if (epoch % report_every == 0 || epoch == 1) {
        int correct = 0;
        for (int i = 0; i < subset_size; ++i) {
          Eigen::Index predicted_label;
          Y_pred.row(i).maxCoeff(&predicted_label);

          Eigen::Index true_label;
          Y_true.row(i).maxCoeff(&true_label);

          if (predicted_label == true_label) {
            correct++;
          }
        }
        float accuracy = (static_cast<float>(correct) / subset_size) * 100.0f;

        std::cout << "Epoch " << std::setw(3) << epoch
                  << " | Loss: " << std::fixed << std::setprecision(4) << loss
                  << " | Accuracy: " << std::setprecision(1) << accuracy
                  << "%\n";
      }
    }
  }
};

} // namespace nn
