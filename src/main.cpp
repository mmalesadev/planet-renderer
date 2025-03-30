#include "vulkan_engine.h"
#include <spdlog/spdlog.h>

int main() {
  spdlog::info("Starting Planet Renderer");
  VulkanEngine engine;
  engine.Init();
  engine.Run();
  engine.Destroy();
  return 0;
}
