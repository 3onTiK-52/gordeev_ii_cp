#include <gtest/gtest.h>
#include <nn/core/activation_function.hpp>
#include <nn/layers/activation_layer.hpp>

TEST(ActivationLayerTest, WorksLikeLayer) {
  // Создаем слой, передав в него ReLU
  nn::ActivationLayer layer{nn::ReLU()};

  // Входные данные
  nn::Matrix X(1, 4);
  X << -5.0f, 0.0f, 3.0f, 10.0f;

  // Проверяем Forward (отрицательные должны стать нулями)
  nn::Matrix expected_Y(1, 4);
  expected_Y << 0.0f, 0.0f, 3.0f, 10.0f;
  EXPECT_TRUE(layer.forward(X).isApprox(expected_Y));

  // Проверяем Backward
  nn::Matrix grad_output(1, 4);
  grad_output << 2.0f, 2.0f, 2.0f, 2.0f;

  // Производная ReLU для -5 и 0 равна 0. Для 3 и 10 равна 1.
  // Значит ошибка пройдет только там, где были положительные числа.
  nn::Matrix expected_dX(1, 4);
  expected_dX << 0.0f, 0.0f, 2.0f, 2.0f;

  EXPECT_TRUE(layer.backward(grad_output).isApprox(expected_dX));
}