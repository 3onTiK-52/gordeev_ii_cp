#include <algorithm>
#include <iostream>
#include <nn/core/activation_function.hpp>
#include <nn/data/dataloader.hpp>
#include <nn/layers/activation_layer.hpp>
#include <nn/layers/linear_layer.hpp>
#include <nn/losses/cross_entropy.hpp>
#include <nn/optimizers/sgd.hpp>
#include <nn/sequential.hpp>
#include <nn/utils/trainer.hpp>

int main() {
  std::cout << "=== MNIST Digit Recognizer (Trainer Edition) ===\n\n";

  // 1. Скачивание(загрузка) данных
  std::string train_path =
      "/home/zontik/cp_2_term/gordeev_ii_cp/mnist_train.csv";

  std::cout << "Loading dataset...\n";
  nn::Dataset full_data;
  try {
    full_data = nn::DataLoader::load_csv(train_path, 0, true);
  } catch (const std::exception &e) {
    std::cerr << "Error loading data: " << e.what() << "\n";
    return 1;
  }

  int subset_size = std::min(1000, (int)full_data.features.rows());
  nn::Matrix X_raw = full_data.features.topRows(subset_size);
  nn::Matrix Y_raw = full_data.targets.topRows(subset_size);

  std::cout << "Using subset of " << subset_size << " images for training.\n\n";

  // 2. Считывание данных
  nn::Matrix X = X_raw / 255.0f;

  nn::Matrix Y_one_hot = nn::Matrix::Zero(subset_size, 10);
  for (int i = 0; i < subset_size; ++i) {
    int label = static_cast<int>(Y_raw(i, 0));
    if (label >= 0 && label <= 9) {
      Y_one_hot(i, label) = 1.0f;
    }
  }

  // 3. Строение нейросети (слои)
  nn::Sequential model;
  model.add(std::make_unique<nn::LinearLayer>(784, 128));
  model.add(std::make_unique<nn::ActivationLayer>(nn::ReLU()));
  model.add(std::make_unique<nn::LinearLayer>(128, 10));

  // 4. Инициализация методов (Loss, Optimizer)
  nn::CrossEntropy criterion;
  nn::SGD optimizer(0.1f);

  // 5. Обучение
  nn::Trainer trainer(model, criterion, optimizer);

  std::cout << "Starting training via Trainer class...\n";
  trainer.train(X, Y_one_hot, 300, 50);

  // 6. Результат
  std::cout << "\n=== Final Predictions (First 5 images) ===\n";
  nn::Matrix sample_X = X.topRows(5);
  nn::Matrix sample_Y_pred = model.forward(sample_X);

  for (int i = 0; i < 5; ++i) {
    Eigen::Index predicted_label;
    sample_Y_pred.row(i).maxCoeff(&predicted_label);

    std::cout << "Image " << i + 1 << " -> Actual digit: " << Y_raw(i, 0)
              << " | Network guessed: " << predicted_label << "\n";
  }

  return 0;
}
