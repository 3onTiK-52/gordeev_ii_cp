#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <nn/data/dataloader.hpp>

TEST(DataLoaderTest, ParsesCSVCorrectly) {
  std::string filename = "dummy_dataset.csv";

  // 1. Создаем тестовый CSV файл
  std::ofstream file(filename);
  file << "Area,Rooms,Price\n";
  file << "50.0,2.0,5000000.0\n";
  file << "70.5,3.0,7500000.0\n";
  file << "30.0,1.0,3000000.0\n";
  file.close();

  // 2. Пытаемся загрузить (целевая колонка - индекс 2, т.е. 'Price')
  nn::Dataset data = nn::DataLoader::load_csv(filename, 2, true);

  // 3. Проверяем размеры матриц
  EXPECT_EQ(data.features.rows(), 3);
  EXPECT_EQ(data.features.cols(), 2);

  EXPECT_EQ(data.targets.rows(), 3);
  EXPECT_EQ(data.targets.cols(), 1);

  // 4. Проверяем конкретные значения (второй дом)
  EXPECT_FLOAT_EQ(data.features(1, 0), 70.5f);
  EXPECT_FLOAT_EQ(data.features(1, 1), 3.0f);
  EXPECT_FLOAT_EQ(data.targets(1, 0), 7500000.0f);

  // 5. Удаляем созданный файл
  std::remove(filename.c_str());
}
