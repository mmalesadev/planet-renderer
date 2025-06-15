cmake_minimum_required(VERSION 4.0)

project(planet-renderer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# System packages
find_package(Vulkan REQUIRED)

# Sources
file(GLOB SRC CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")
add_executable(${PROJECT_NAME} ${SRC})

# Shaders
file(GLOB SHADERS CONFIGURE_DEPENDS "src/shaders/*.vert" "src/shaders/*.frag")
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})
foreach(SHADER ${SHADERS})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SPIRV_OUTPUT "${SHADER_OUTPUT_DIR}/${FILE_NAME}.spv")
    add_custom_command(OUTPUT ${SPIRV_OUTPUT} COMMAND glslc ${SHADER} -o ${SPIRV_OUTPUT} DEPENDS ${SHADER} COMMENT "Compiling shader ${FILE_NAME}" VERBATIM)
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUTPUT})
endforeach()
add_custom_target(compile_shaders ALL DEPENDS ${SPIRV_OUTPUTS})

# Libraries built from source
set(SDL_TEST_LIBRARY OFF)
add_subdirectory(libs/SDL)
add_subdirectory(libs/spdlog)
add_subdirectory(libs/glm)

target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan SDL3::SDL3 glm::glm spdlog::spdlog)

# Copy Windows DLLs
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL3::SDL3>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()




