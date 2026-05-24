#include <cstdlib>
#include <gtest/gtest.h>
#include <nn/core/activation_function.hpp>
#include <nn/layers/activation_layer.hpp>
#include <nn/layers/linear_layer.hpp>
#include <nn/losses/mse.hpp>
#include <nn/optimizers/sgd.hpp>
#include <nn/sequential.hpp>

TEST(IntegrationTest, NetworkLearnsXOR) {
  std::srand(42);

  nn::Sequential model;
  model.add(std::make_unique<nn::LinearLayer>(2, 4));
  model.add(std::make_unique<nn::ActivationLayer>(nn::ReLU()));
  model.add(std::make_unique<nn::LinearLayer>(4, 1));

  nn::MSE loss_f;
  nn::SGD optimizer(0.1f);

  // Данные XOR
  nn::Matrix X(4, 2);
  X << 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f;

  nn::Matrix Y_true(4, 1);
  Y_true << 0.0f, 1.0f, 1.0f, 0.0f;

  int epochs = 600;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    nn::Matrix Y_pred = model.forward(X);
    nn::Matrix grad = loss_f.backward(Y_pred, Y_true);
    model.backward(grad);
    model.update(optimizer);
  }

  // Делаем финальное предсказание
  nn::Matrix final_predictions = model.forward(X);

  // Проверяем, что сеть действительно выучила логику XOR
  EXPECT_LT(final_predictions(0, 0), 0.001f); // 0 XOR 0 = 0
  EXPECT_GT(final_predictions(1, 0), 0.999f); // 0 XOR 1 = 1
  EXPECT_GT(final_predictions(2, 0), 0.999f); // 1 XOR 0 = 1
  EXPECT_LT(final_predictions(3, 0), 0.001f); // 1 XOR 1 = 0
}
