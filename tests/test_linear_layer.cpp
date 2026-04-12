#include <gtest/gtest.h>
#include <nn/layers/linear_layer.hpp>

// Тестируем Forward
TEST(LinearLayerTest, ForwardPass) {
  nn::LinearLayer layer(2, 2);

  // 1. Задаем веса W и сдвиги b
  nn::Matrix W(2, 2);
  W << 0.1f, 0.2f, 0.3f, 0.4f;
  nn::Matrix b(1, 2);
  b << 0.5f, 0.6f;
  layer.set_weights(W);
  layer.set_biases(b);

  nn::Matrix X(2, 2);
  X << 1.0f, 2.0f, 3.0f, 4.0f;

  // 2. Ожидаемый результат
  nn::Matrix expected_Y(2, 2);
  expected_Y << 1.2f, 1.6f, 2.0f, 2.8f;

  // Сравниваем результаты (считаем 1e-4 допустимой погрешностью)
  nn::Matrix Y = layer.forward(X);
  EXPECT_TRUE(Y.isApprox(expected_Y, 1e-4));
}

// Тестируем Backward
TEST(LinearLayerTest, BackwardPass) {
  nn::LinearLayer layer(2, 2);

  nn::Matrix W(2, 2);
  W << 0.1f, 0.2f, 0.3f, 0.4f;
  layer.set_weights(W);

  // Для backward нам нужен кэш X, поэтому сначала делаем forward
  nn::Matrix X(2, 2);
  X << 1.0f, 2.0f, 3.0f, 4.0f;
  layer.forward(X);

  nn::Matrix grad_output(2, 2);
  grad_output << 1.0f, 1.0f, 1.0f, 1.0f;

  nn::Matrix dX = layer.backward(grad_output);

  // Ожидаемые градиенты (dW = X^T * dY, db = sum(dY), dX = dY * W^T)
  nn::Matrix expected_dW(2, 2);
  expected_dW << 4.0f, 4.0f, 6.0f, 6.0f;
  nn::Matrix expected_db(1, 2);
  expected_db << 2.0f, 2.0f;
  nn::Matrix expected_dX(2, 2);
  expected_dX << 0.3f, 0.7f, 0.3f, 0.7f;

  // Проверяем
  EXPECT_TRUE(layer.get_grad_weights().isApprox(expected_dW, 1e-4));
  EXPECT_TRUE(layer.get_grad_biases().isApprox(expected_db, 1e-4));
  EXPECT_TRUE(dX.isApprox(expected_dX, 1e-4));
}
