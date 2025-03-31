#include <SDL3/SDL.h>
#include <optional>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

class VulkanEngine {
public:
  void Init();
  void Run();
  void Destroy();

private:
  struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> presentation_family;

    bool IsComplete() {
      return graphics_family.has_value() && presentation_family.has_value();
    }
  };

  struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
  };

  void InitSDL();
  void InitVulkanInstance();
  void CreateSurface();
  void ListAvailableExtensions() const;

  // Validation Layers
  bool CheckValidationLayerSupport();
  void SetupDebugMessenger();

  // Devices
  void PickPhysicalDevice();
  bool IsDeviceSuitable(VkPhysicalDevice device);
  bool CheckDeviceExtensionSupport(VkPhysicalDevice device);
  void CreateLogicalDevice();
  QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device);

  // Swap chain
  void CreateSwapChain();
  VkSurfaceFormatKHR ChooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &available_formats);
  VkPresentModeKHR ChooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &available_present_modes);
  VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);
  SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device);

  // Image views
  void CreateImageViews();

  // Graphics pipeline
  void CreateGraphicsPipeline();
  std::optional<VkShaderModule>
  CreateShaderModule(const std::vector<char> &code);
  void CreateRenderPass();

  // Framebuffers
  void CreateFramebuffers();

  // Commands
  void CreateCommandPool();
  void CreateCommandBuffer();
  void RecordCommandBuffer(VkCommandBuffer command_buffer,
                           uint32_t image_index);

  // Drawing
  void CreateSyncObjects();
  void DrawFrame();

  VkInstance instance_;
  VkDebugUtilsMessengerEXT debug_messenger_;
  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_;
  VkQueue graphics_queue_;
  VkSurfaceKHR surface_;
  VkQueue presentation_queue_;
  VkSwapchainKHR swap_chain_;
  std::vector<VkImage> swap_chain_images_;
  VkFormat swap_chain_image_format_;
  VkExtent2D swap_chain_extent_;
  std::vector<VkImageView> swap_chain_image_views_;
  VkRenderPass render_pass_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;
  std::vector<VkFramebuffer> swap_chain_framebuffers_;
  VkCommandPool command_pool_;
  VkCommandBuffer command_buffer_;
  VkSemaphore image_available_semaphore_;
  VkSemaphore render_finished_semaphore_;
  VkFence in_flight_fence_;

  bool running_;
  SDL_Window *window_;
};
