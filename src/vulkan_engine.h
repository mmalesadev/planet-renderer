#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

class VulkanEngine {
public:
  void Init();
  void Run();
  void Destroy();

private:
  void InitSDL();
  void InitVulkanInstance();
  void ListAvailableExtensions() const;

  // Validation Layers
  bool CheckValidationLayerSupport();
  void SetupDebugMessenger();

  // Devices
  void PickPhysicalDevice();
  bool IsDeviceSuitable(VkPhysicalDevice device);
  void CreateLogicalDevice();

  VkInstance instance_;
  VkDebugUtilsMessengerEXT debug_messenger_;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_;
  VkQueue graphics_queue_;
  bool running_;
  SDL_Window *window_;
  SDL_Renderer *fake_renderer_;
};
