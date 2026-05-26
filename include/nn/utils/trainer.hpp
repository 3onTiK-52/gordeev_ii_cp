#pragma once
#include "../losses/loss.hpp"
#include "../optimizers/optimizer.hpp"
#include "../sequential.hpp"
#include <algorithm>
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
             int batch_size = 64) {
    int n_samples = X.rows();

    for (int epoch = 1; epoch <= epochs; ++epoch) {
      float total_loss = 0.0f;
      int correct_predictions = 0;

      // Разбиваем весь датасет на маленькие батчи
      for (int start_idx = 0; start_idx < n_samples; start_idx += batch_size) {

        int current_batch_size = std::min(batch_size, n_samples - start_idx);

        Matrix X_batch = X.middleRows(start_idx, current_batch_size);
        Matrix Y_batch = Y_true.middleRows(start_idx, current_batch_size);

        // 1. Прямой проход для батча
        Matrix Y_pred = model_.forward(X_batch);

        // 2. Ошибка батча
        float batch_loss = criterion_.forward(Y_pred, Y_batch);
        total_loss += batch_loss * current_batch_size;

        // 3. Backward & update
        Matrix grad = criterion_.backward(Y_pred, Y_batch);
        model_.backward(grad);
        model_.update(optimizer_);

        // 4. Подсчет точности
        for (int i = 0; i < current_batch_size; ++i) {
          Eigen::Index pred_idx, true_idx;
          Y_pred.row(i).maxCoeff(&pred_idx);
          Y_batch.row(i).maxCoeff(&true_idx);
          if (pred_idx == true_idx)
            correct_predictions++;
        }
      }

      // Выводим статистику для каждой эпохи
      float epoch_loss = total_loss / n_samples;
      float epoch_accuracy =
          (static_cast<float>(correct_predictions) / n_samples) * 100.0f;

      std::cout << "Epoch " << std::setw(3) << epoch
                << " | Loss: " << std::fixed << std::setprecision(4)
                << epoch_loss << " | Train Accuracy: " << std::setprecision(2)
                << epoch_accuracy << "%\n";
    }
  }
};

} // namespace nn
