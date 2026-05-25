#include <gtest/gtest.h>
#include <nn/losses/cross_entropy.hpp>

TEST(LossesTest, CrossEntropyForwardBackward) {
  nn::CrossEntropy ce;

  // Предсказания сети (сумма не равна 1)
  nn::Matrix Y_pred(1, 3);
  Y_pred << 1.0f, 2.0f, 1.0f;

  // Правильный ответ (это класс с индексом 1)
  nn::Matrix Y_true(1, 3);
  Y_true << 0.0f, 1.0f, 0.0f;

  // Ручной просчет Softmax даст вероятности ~ [0.21, 0.58, 0.21]
  // Кросс-Энтропия: -log(0.58) ≈ 0.5514
  float loss = ce.forward(Y_pred, Y_true);
  EXPECT_NEAR(loss, 0.5514f, 1e-3);

  // Формула градиента: (Probs - True) / N
  // Ожидаем: [0.2119, 0.5761 - 1.0, 0.2119]
  nn::Matrix expected_grad(1, 3);
  expected_grad << 0.2119f, -0.4239f, 0.2119f;

  nn::Matrix grad = ce.backward(Y_pred, Y_true);
  EXPECT_TRUE(grad.isApprox(expected_grad, 1e-3));
}
