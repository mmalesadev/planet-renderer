cmake_minimum_required(VERSION 4.0)

project(planet-renderer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)

find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED)

file(GLOB SRC CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")

add_executable(${PROJECT_NAME} ${SRC})

set(SDL_TEST_LIBRARY OFF)
# add_subdirectory(libs/SDL)
add_subdirectory(libs/spdlog)
add_subdirectory(libs/glm)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan SDL3::SDL3 glm::glm spdlog::spdlog)

