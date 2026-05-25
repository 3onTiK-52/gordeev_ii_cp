#pragma once
#include "../core/types.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {

// Вспомогательная структура, чтобы возвращать сразу две матрицы
struct Dataset {
  Matrix features; // Матрица X (все колонки, кроме целевой)
  Matrix targets;  // Матрица Y (целевая колонка)
};

class DataLoader {
public:
  static Dataset load_csv(const std::string &filepath, int target_col_idx = -1,
                          bool has_header = true) {
    std::ifstream file(filepath.c_str());
    if (!file.is_open()) {
      throw std::runtime_error("Не удалось открыть файл: " + filepath);
    }

    std::vector<std::vector<float>> data;
    std::string line;

    // Пропускаем заголовок
    if (has_header) {
      std::getline(file, line);
    }

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      std::vector<float> row;
      std::stringstream ss(line);
      std::string cell;

      while (std::getline(ss, cell, ',')) {
        try {
          row.push_back(std::stof(cell));
        } catch (...) {
          // Если ячейка пустая или содержит текст, ставим 0.0
          row.push_back(0.0f);
        }
      }
      if (!row.empty()) {
        data.push_back(row);
      }
    }

    if (data.empty()) {
      throw std::runtime_error("Файл пуст или не содержит корректных данных");
    }

    int rows = data.size();
    int cols = data[0].size();

    if (target_col_idx < 0) {
      target_col_idx = cols - 1;
    }

    Dataset dataset;
    dataset.features = Matrix(rows, cols - 1);
    dataset.targets = Matrix(rows, 1);

    for (int i = 0; i < rows; ++i) {
      int feature_idx = 0;
      for (int j = 0; j < cols; ++j) {
        if (j == target_col_idx) {
          dataset.targets(i, 0) = data[i][j];
        } else {
          dataset.features(i, feature_idx++) = data[i][j];
        }
      }
    }

    return dataset;
  }
};

} // namespace nn
