#include <gtest/gtest.h>
#include <nn/optimizers/sgd.hpp>

TEST(OptimizersTest, SGDUpdateWithoutWeightDecay) {
  nn::SGD optimizer(0.1f); // lr = 0.1

  // Начальные веса: 1.0
  nn::Matrix W(1, 1);
  W << 1.0f;

  // Градиент: 2.0 (сеть говорит, что мы ошиблись "вверх" на 2.0)
  nn::Matrix dW(1, 1);
  dW << 2.0f;

  // Делаем шаг оптимизации
  optimizer.update(W, dW);

  // Ожидаем: W_new = 1.0 - 0.1 * 2.0 = 0.8
  EXPECT_NEAR(W(0, 0), 0.8f, 1e-5);
}

TEST(OptimizersTest, SGDUpdateWithWeightDecay) {
  nn::SGD optimizer(0.1f, 0.5f); // lr = 0.1, weight_decay = 0.5

  nn::Matrix W(1, 1);
  W << 1.0f;
  nn::Matrix dW(1, 1);
  dW << 2.0f;

  optimizer.update(W, dW);

  // Ожидаем: W_new = 1.0 - 0.1 * (2.0 + 0.5 * 1.0) = 1.0 - 0.1 * 2.5 = 0.75
  EXPECT_NEAR(W(0, 0), 0.75f, 1e-5);
}
