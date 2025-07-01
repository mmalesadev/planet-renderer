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
  void InitImGui();

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
  void CleanupSwapChain();
  void RecreateSwapChain();

  // Image views
  void CreateImageViews();

  // Graphics pipeline
  void CreateDescriptorSetLayout();
  void CreateGraphicsPipeline();
  std::optional<VkShaderModule>
  CreateShaderModule(const std::vector<char> &code);
  void CreateRenderPass();

  // Framebuffers
  void CreateFramebuffers();

  // Buffers: Vertex, Index, Uniform
  void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &buffer_memory);
  void CopyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size);
  void CreateVertexBuffer();
  uint32_t FindMemoryType(uint32_t type_filter,
                          VkMemoryPropertyFlags properties);
  void CreateIndexBuffer();
  void CreateUniformBuffers();
  void UpdateUniformBuffer(uint32_t current_image);
  void CreateDescriptorPool();
  void CreateDescriptorSets();

  // Depth buffer
  void CreateDepthResources();
  std::optional<VkFormat>
  FindSupportedFormat(const std::vector<VkFormat> &candidates,
                      VkImageTiling tiling, VkFormatFeatureFlags features);
  std::optional<VkFormat> FindDepthFormat();
  std::optional<VkImageView> CreateImageView(VkImage image, VkFormat format,
                                             VkImageAspectFlags aspect_flags);
  void CreateImage(uint32_t width, uint32_t height,
                   VkSampleCountFlagBits num_samples, VkFormat format,
                   VkImageTiling tiling, VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &image_memory);
  void TransitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout old_layout,
                             VkImageLayout new_layout);

  // Commands
  void CreateCommandPool();
  void CreateCommandBuffer();
  void RecordCommandBuffer(VkCommandBuffer command_buffer,
                           uint32_t image_index);
  VkCommandBuffer BeginSingleTimeCommands();
  void EndSingleTimeCommands(VkCommandBuffer command_buffer);

  // Multisapmpling
  VkSampleCountFlagBits GetMaxUsableSampleCount() const;
  void CreateColorResources();

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
  VkDescriptorSetLayout descriptor_set_layout_;
  VkPipelineLayout pipeline_layout_;
  VkPipeline graphics_pipeline_;
  std::vector<VkFramebuffer> swap_chain_framebuffers_;
  VkCommandPool command_pool_;
  // NOTE:
  // You should allocate multiple resources like buffers from a single memory
  // allocation.
  // NOTE:
  // Driver developers recommend that you also store multiple buffers, like
  // the vertex and index buffer, into a single VkBuffer and use offsets in
  // commands like vkCmdBindVertexBuffers. The advantage is that your data is
  // more cache friendly in that case, because it's closer together. It is even
  // possible to reuse the same chunk of memory for multiple resources if they
  // are not used during the same render operations, provided that their data is
  // refreshed, of course. This is known as aliasing and some Vulkan functions
  // have explicit flags to specify that you want to do this.
  VkBuffer vertex_buffer_;
  VkDeviceMemory vertex_buffer_memory_;
  VkBuffer index_buffer_;
  VkDeviceMemory index_buffer_memory_;
  std::vector<VkBuffer> uniform_buffers_;
  std::vector<VkDeviceMemory> uniform_buffers_memory_;
  std::vector<void *> uniform_buffers_mapped_;
  VkDescriptorPool descriptor_pool_;
  std::vector<VkDescriptorSet> descriptor_sets_;
  VkImage depth_image_;
  VkDeviceMemory depth_image_memory_;
  VkImageView depth_image_view_;
  std::vector<VkCommandBuffer> command_buffers_;
  std::vector<VkSemaphore> image_available_semaphores_;
  std::vector<VkSemaphore> render_finished_semaphores_;
  std::vector<VkFence> in_flight_fences_;
  VkSampleCountFlagBits msaa_samples_ = VK_SAMPLE_COUNT_1_BIT;
  VkImage color_image_;
  VkDeviceMemory color_image_memory_;
  VkImageView color_image_view_;
  bool resize_requested_ = false;
  bool freeze_rendering_ = false;

  VkDescriptorPool imgui_descriptor_pool_;

  SDL_Window *window_;

  uint32_t current_frame_ = 0;
  bool running_;
};
