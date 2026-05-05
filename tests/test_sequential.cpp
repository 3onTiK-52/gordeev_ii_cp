#include <gtest/gtest.h>
#include <nn/core/activation_function.hpp>
#include <nn/layers/activation_layer.hpp>
#include <nn/layers/linear_layer.hpp>
#include <nn/sequential.hpp>

TEST(SequentialTest, FullForwardPass) {
  nn::Sequential model;

  // Делаем два слоя: линейный + активация
  model.add(std::make_unique<nn::LinearLayer>(2, 2));
  model.add(std::make_unique<nn::ActivationLayer>(nn::ReLU()));

  nn::Matrix X(1, 2);
  X << 2.0f, -1.0f;

  // Проверка, что модель запускается и возвращает данные
  nn::Matrix Y = model.forward(X);

  // Проверка сбоев с размером выходных данных
  EXPECT_EQ(Y.rows(), 1);
  EXPECT_EQ(Y.cols(), 2);
}

TEST(SequentialTest, ExactValuesPass) {
  nn::Sequential model;

  // 1. Создаем первый линейный слой
  auto linear1 = std::make_unique<nn::LinearLayer>(2, 2);
  auto *lin1_ptr = linear1.get();

  // 2. Нелинейный слой
  auto activation = std::make_unique<nn::ActivationLayer>(nn::ReLU());

  // 3. Второй линейный слой
  auto linear2 = std::make_unique<nn::LinearLayer>(2, 1);
  auto *lin2_ptr = linear2.get();

  // Передаем владение объектами внутрь модели
  model.add(std::move(linear1));
  model.add(std::move(activation));
  model.add(std::move(linear2));

  // Настраиваем веса
  nn::Matrix W1(2, 2);
  W1 << 1.0f, -1.0f, -1.0f, 1.0f;
  nn::Matrix b1(1, 2);
  b1 << 0.0f, 0.0f;
  lin1_ptr->set_weights(W1);
  lin1_ptr->set_biases(b1);

  nn::Matrix W2(2, 1);
  W2 << 2.0f, 2.0f;
  nn::Matrix b2(1, 1);
  b2 << -1.0f;
  lin2_ptr->set_weights(W2);
  lin2_ptr->set_biases(b2);

  // Входные данные
  nn::Matrix X(1, 2);
  X << 2.0f, 1.0f;

  // Считаем руками
  // Слой 1 (Linear): X * W1 + b1 = [2, 1] * [[1, -1], [-1, 1]] = [1.0, -1.0]
  // Слой 2 (ReLU): max(0, [1.0, -1.0]) = [1.0, 0.0]
  // Слой 3 (Linear): [1.0, 0.0] * [[2], [2]] - 1.0 = (1*2 + 0*2) - 1.0 = 1.0

  nn::Matrix expected_Y(1, 1);
  expected_Y << 1.0f;

  // Сверяем результат работы сети с нашими расчетами
  nn::Matrix Y = model.forward(X);
  EXPECT_TRUE(Y.isApprox(expected_Y, 1e-4));
}
