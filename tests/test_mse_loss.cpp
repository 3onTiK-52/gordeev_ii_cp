#include <gtest/gtest.h>
#include <nn/losses/mse.hpp>

TEST(LossesTest, MSEForwardAndBackward) {
  nn::MSE mse;

  // Предсказания нашей сети(условно)
  nn::Matrix Y_pred(2, 1);
  Y_pred << 1.0f, 2.0f;

  // Таргеты
  nn::Matrix Y_true(2, 1);
  Y_true << 1.0f, 4.0f;

  // Формула: ( (1-1)^2 + (2-4)^2 ) / 2 = (0 + 4) / 2 = 2.0
  float loss = mse.forward(Y_pred, Y_true);
  EXPECT_NEAR(loss, 2.0f, 1e-4); // EXPECT_NEAR нужен для сравнения float чисел

  // Формула: 2 * (Y_pred - Y_true) / 2 = Y_pred - Y_true
  // [1-1, 2-4] = [0, -2]
  nn::Matrix expected_grad(2, 1);
  expected_grad << 0.0f, -2.0f;

  nn::Matrix grad = mse.backward(Y_pred, Y_true);
  EXPECT_TRUE(grad.isApprox(expected_grad, 1e-4));
}
