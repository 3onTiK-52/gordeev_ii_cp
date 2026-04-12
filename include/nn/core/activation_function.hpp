#pragma once
#include "AnyMovable.h"
#include <Eigen/Dense>

namespace nn {

using Matrix = Eigen::MatrixXf;

// 1. Что должна уметь делать любая функция активации
template <class TBase> class IActivation : public TBase {
public:
  virtual Matrix forward(const Matrix &input) const = 0;
  virtual Matrix backward(const Matrix &input,
                          const Matrix &grad_output) const = 0;
};

// 2. Класс-реализация (обертка)
template <class TBase, class TObject> class CActivationImpl : public TBase {
  using CBase = TBase;

public:
  using CBase::CBase;

  Matrix forward(const Matrix &input) const override {
    return CBase::Object().forward(input);
  }

  Matrix backward(const Matrix &input,
                  const Matrix &grad_output) const override {
    return CBase::Object().backward(input, grad_output);
  }
};

// 3. Финальный стирающий тип для функций активации
class AnyActivation
    : public NSLibrary::CAnyMovable<IActivation, CActivationImpl> {
  using CBase = NSLibrary::CAnyMovable<IActivation, CActivationImpl>;

public:
  using CBase::CBase;
};

// ===========================================================================
// 1) Функция ReLU: Все значения меньше 0 заменяются на 0, производная 1 если
// элемент > 0, иначе 0.

struct ReLU {
  Matrix forward(const Matrix &input) const { return input.cwiseMax(0.0f); }

  Matrix backward(const Matrix &input, const Matrix &grad_output) const {
    return (input.array() > 0.0f).cast<float>() * grad_output.array();
  }
};

} // namespace nn