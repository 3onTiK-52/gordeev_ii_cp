#include <fstream>
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
  std::cout << "=== mnist.train pipeline ===\n\n";

  // 1. Загрузка обучающего датасета
  std::string train_path =
      "/home/zontik/cp_2_term/gordeev_ii_cp/mnist_train.csv";
  std::cout << "Loading FULL train dataset from " << train_path << "...\n";

  nn::Dataset train_data;
  try {
    train_data = nn::DataLoader::load_csv(train_path, 0, true);
  } catch (const std::exception &e) {
    std::cerr << "Fatal Error: " << e.what() << "\n";
    return 1;
  }

  int n_samples = train_data.features.rows();
  std::cout << "Loaded " << n_samples << " images for training.\n\n";

  // Нормализация 0..1
  nn::Matrix X_train = train_data.features / 255.0f;

  // One-Hot Encoding для всех 60,000 строк
  nn::Matrix Y_train_one_hot = nn::Matrix::Zero(n_samples, 10);
  for (int i = 0; i < n_samples; ++i) {
    int label = static_cast<int>(train_data.targets(i, 0));
    if (label >= 0 && label <= 9)
      Y_train_one_hot(i, label) = 1.0f;
  }

  // 2. Строение сети и обучение
  std::cout << "Building architecture...\n";
  nn::Sequential model;
  model.add(std::make_unique<nn::LinearLayer>(784, 128));
  model.add(std::make_unique<nn::ActivationLayer>(nn::ReLU()));
  model.add(std::make_unique<nn::LinearLayer>(128, 10));

  nn::CrossEntropy criterion;
  nn::SGD optimizer(0.1f);
  nn::Trainer trainer(model, criterion, optimizer);

  // За счет батчей сеть обновляет веса чуть около 1000 раз за эпоху.
  // Поэтому 15-20 эпох вполне должно хватить (update: В целом хватает 10
  // эпох(98.3%), делать 15(99.0%) или 20 можно только при сильной
  // необходимости)
  int epochs = 15;
  int batch_size = 64;

  std::cout << "Starting Mini-Batch Training (Epochs: " << epochs
            << ", Batch size: " << batch_size << ")...\n";
  trainer.train(X_train, Y_train_one_hot, epochs, batch_size);

  // 3. Сохранение весов
  std::string weights_file = "mnist_weights_full.bin";
  std::cout << "\nSaving production weights to " << weights_file << "...\n";
  model.save(weights_file);

  // 4. Проверяем нашу модель на тестовой выборке
  std::string test_path = "/home/zontik/cp_2_term/gordeev_ii_cp/mnist_test.csv";
  std::cout
      << "\n=== Testing Neutral Network in real situation(not at all) ===\n";
  std::cout << "Loading test dataset from " << test_path << "...\n";

  nn::Dataset test_data;
  try {
    test_data = nn::DataLoader::load_csv(test_path, 0, true);
  } catch (const std::exception &e) {
    std::cerr << "Warning: Could not load test file (" << e.what()
              << "). Skipping inference.\n";
    return 0;
  }

  nn::Matrix X_test = test_data.features / 255.0f;
  nn::Matrix Y_test_raw = test_data.targets;
  int test_samples = X_test.rows();
  std::cout << "Evaluating " << test_samples << " unseen images...\n";

  // Прямой проход
  nn::Matrix Y_test_pred = model.forward(X_test);

  // Считаем честную точность
  int correct_predictions = 0;
  std::string submission_file = "submission.csv";
  std::ofstream sub(submission_file);

  if (sub.is_open())
    sub << "ImageId,Label\n";

  for (int i = 0; i < test_samples; ++i) {
    Eigen::Index predicted_label;
    Y_test_pred.row(i).maxCoeff(&predicted_label);

    int true_label = static_cast<int>(Y_test_raw(i, 0));
    if (predicted_label == true_label) {
      correct_predictions++;
    }

    // Заодно пишем в сабмит
    if (sub.is_open()) {
      sub << (i + 1) << "," << predicted_label << "\n";
    }
  }
  if (sub.is_open())
    sub.close();

  float test_accuracy =
      (static_cast<float>(correct_predictions) / test_samples) * 100.0f;

  std::cout << "\n============================================\n";
  std::cout << "FINAL TEST ACCURACY: " << std::fixed << std::setprecision(2)
            << test_accuracy << "%\n";
  std::cout << "============================================\n";
  std::cout << "Predictions also saved to " << submission_file << "!\n";

  return 0;
}
