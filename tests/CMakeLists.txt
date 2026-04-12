# Создаем отдельный исполняемый файл для тестов
add_executable(nn_tests test_activation.cpp test_linear_layer.cpp test_activation_layer.cpp)

# Подключаем к нему нашу библиотеку нейросети и библиотеку GTest
target_link_libraries(nn_tests PRIVATE nn_lib GTest::gtest_main)

# Сообщаем CMake, чтобы он нашел все тесты в этом файле
include(GoogleTest)
gtest_discover_tests(nn_tests)