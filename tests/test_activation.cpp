#include <gtest/gtest.h>
#include <nn/core/activation_function.hpp>

// Макрос TEST принимает два аргумента: (НазваниеГруппыТестов,
// НазваниеКонкретногоТеста)
TEST(ActivationTest, ReLUForward) {
  nn::AnyActivation relu = nn::ReLU();

  nn::Matrix X(1, 4);
  X << -3.0f, 0.0f, 2.5f, 10.0f;

  nn::Matrix expected_Y(1, 4);
  expected_Y << 0.0f, 0.0f, 2.5f, 10.0f;

  nn::Matrix Y = relu->forward(X);

  // Метод .isApprox() - это функция библиотеки Eigen.
  // Она сравнивает две матрицы с учетом эпсилон погрешностей
  EXPECT_TRUE(Y.isApprox(expected_Y));
}

TEST(ActivationTest, ReLUBackward) {
  nn::AnyActivation relu = nn::ReLU();

  nn::Matrix X(1, 4);
  X << -3.0f, 0.0f, 2.5f, 10.0f;

  nn::Matrix grad_output(1, 4);
  grad_output << 5.0f, 5.0f, 5.0f, 5.0f;

  nn::Matrix expected_dX(1, 4);
  expected_dX << 0.0f, 0.0f, 5.0f, 5.0f;

  nn::Matrix dX = relu->backward(X, grad_output);

  EXPECT_TRUE(dX.isApprox(expected_dX));
}
