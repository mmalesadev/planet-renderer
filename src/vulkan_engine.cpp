#include "vulkan_engine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <queue>
#include <set>
#include <spdlog/spdlog.h>
#include <utility>
#include <vulkan/vulkan_core.h>

namespace {

const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// TODO: Enable validation layers only on debug builds
const bool kEnableValidationLayers = true;

VKAPI_ATTR VkBool32 VKAPI_CALL
DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
              VkDebugUtilsMessageTypeFlagsEXT message_type,
              const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
              void *p_user_data) {
  if (message_severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    spdlog::warn("Validation layer: {}", p_callback_data->pMessage);
  } else if (message_severity >=
             VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    spdlog::error("Validation layer: {}", p_callback_data->pMessage);
  }
  return VK_FALSE;
}

} // namespace

void VulkanEngine::Init() {
  // TODO: Check if everything initialized correctly
  InitSDL();
  InitVulkanInstance();
  SetupDebugMessenger();
  CreateSurface();
  PickPhysicalDevice();
  CreateLogicalDevice();
  CreateSwapChain();
  CreateImageViews();
}

void VulkanEngine::Run() {
  SDL_Event event;
  while (running_) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT) {
        running_ = false;
      }
    }

    // Slow down to 60 FPS
    SDL_Delay(16);
  }
}

void VulkanEngine::Destroy() {
  for (auto image_view : swap_chain_image_views_) {
    vkDestroyImageView(device_, image_view, nullptr);
  }
  vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
  vkDestroyDevice(device_, nullptr);

  if (kEnableValidationLayers) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance_, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
      func(instance_, debug_messenger_, nullptr);
    }
  }

  vkDestroySurfaceKHR(instance_, surface_, nullptr);
  vkDestroyInstance(instance_, nullptr);
  SDL_DestroyWindow(window_);
  SDL_Quit();
}

void VulkanEngine::InitSDL() {
  SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "wayland");
  SDL_SetAppMetadata("Planet Renderer", "0.0.1", "com.example.planet_renderer");
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    spdlog::error("Failed to initialize SDL: {}", SDL_GetError());
    return;
  }

  SDL_WindowFlags window_flags =
      (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                        SDL_WINDOW_HIGH_PIXEL_DENSITY);
  window_ = SDL_CreateWindow("PlanetRenderer", 800, 600, window_flags);

  if (!window_) {
    spdlog::error("Failed to create window_: {}", SDL_GetError());
    SDL_Quit();
    return;
  }

  running_ = true;
  return;
}

void VulkanEngine::InitVulkanInstance() {
  if (kEnableValidationLayers && !CheckValidationLayerSupport()) {
    spdlog::error("Validation layers requested but not available.");
    return;
  }

  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Planet Renderer";
  app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 1);
  app_info.pEngineName = "Planet Renderer Engine";
  app_info.engineVersion = VK_MAKE_VERSION(0, 0, 1);
  app_info.apiVersion = VK_API_VERSION_1_4;
  app_info.pNext = nullptr;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  uint32_t sdl_extension_count = 0;
  const char *const *sdl_extensions;
  sdl_extensions = SDL_Vulkan_GetInstanceExtensions(&sdl_extension_count);
  std::vector<const char *> extensions(sdl_extensions,
                                       sdl_extensions + sdl_extension_count);
  if (kEnableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();
  for (const auto &e : extensions) {
    spdlog::info("Extension name: {}", e);
  }
  create_info.enabledLayerCount = 0;

  if (kEnableValidationLayers) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    create_info.ppEnabledLayerNames = kValidationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }

  if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
    spdlog::error("Failed to create Vulkan instance");
    return;
  }

  // ListAvailableExtensions();
}

void VulkanEngine::CreateSurface() {
  if (!SDL_Vulkan_CreateSurface(window_, instance_, nullptr, &surface_)) {
    spdlog::error("Failed to create window surface: {}", SDL_GetError());
    return;
  }
}

void VulkanEngine::ListAvailableExtensions() const {
  // Extensions from this list must be loaded to work
  uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extensions(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count,
                                         extensions.data());
  spdlog::info("Available extensions:");
  for (const auto &extension : extensions) {
    spdlog::info("{}", extension.extensionName);
  }
}

bool VulkanEngine::CheckValidationLayerSupport() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
  for (const char *layer_name : kValidationLayers) {
    bool layer_found = false;
    for (const auto &layer_properties : available_layers) {
      if (strcmp(layer_name, layer_properties.layerName) == 0) {
        layer_found = true;
        break;
      }
    }
    if (!layer_found) {
      return false;
    }
  }
  return true;
}

void VulkanEngine::SetupDebugMessenger() {
  if (!kEnableValidationLayers) {
    return;
  }

  VkDebugUtilsMessengerCreateInfoEXT create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info.pfnUserCallback = DebugCallback;

  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance_, "vkCreateDebugUtilsMessengerEXT");
  VkResult result;
  if (func != nullptr) {
    result = func(instance_, &create_info, nullptr, &debug_messenger_);
  } else {
    result = VK_ERROR_EXTENSION_NOT_PRESENT;
  }

  if (result != VK_SUCCESS) {
    spdlog::error("Failed to set up debug messenger.");
    return;
  }
}

void VulkanEngine::PickPhysicalDevice() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  if (device_count == 0) {
    spdlog::error("Failed to find GPUs with Vulkan support.");
    return;
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

  for (const auto &device : devices) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);

    if (IsDeviceSuitable(device)) {
      physical_device_ = device;
      break;
    }
  }

  if (physical_device_ == VK_NULL_HANDLE) {
    spdlog::error("Failed to find suitable GPU.");
    return;
  }
}

bool VulkanEngine::IsDeviceSuitable(VkPhysicalDevice device) {
  QueueFamilyIndices indices = FindQueueFamilies(device);
  bool extensions_supported = CheckDeviceExtensionSupport(device);
  bool swap_chain_adequate = false;
  if (extensions_supported) {
    SwapChainSupportDetails swap_chain_support = QuerySwapChainSupport(device);
    swap_chain_adequate = !swap_chain_support.formats.empty() &&
                          !swap_chain_support.present_modes.empty();
  }
  return indices.IsComplete() && extensions_supported && swap_chain_adequate;
}

bool VulkanEngine::CheckDeviceExtensionSupport(VkPhysicalDevice device) {
  uint32_t extension_count;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       nullptr);

  std::vector<VkExtensionProperties> available_extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count,
                                       available_extensions.data());
  std::set<std::string> required_extensions(kDeviceExtensions.begin(),
                                            kDeviceExtensions.end());
  for (const auto &extension : available_extensions) {
    required_extensions.erase(extension.extensionName);
  }

  return required_extensions.empty();
}

void VulkanEngine::CreateLogicalDevice() {
  QueueFamilyIndices indices = FindQueueFamilies(physical_device_);
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_queue_families = {
      indices.graphics_family.value(), indices.presentation_family.value()};

  float queue_priority = 1.0f;

  for (uint32_t queue_family : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = indices.graphics_family.value();
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }
  VkPhysicalDeviceFeatures device_features{};

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.queueCreateInfoCount =
      static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;

  create_info.enabledExtensionCount =
      static_cast<uint32_t>(kDeviceExtensions.size());
  create_info.ppEnabledExtensionNames = kDeviceExtensions.data();

  if (kEnableValidationLayers) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    create_info.ppEnabledLayerNames = kValidationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }

  if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) !=
      VK_SUCCESS) {
    spdlog::error("Failed to create logical device.");
    return;
  }

  vkGetDeviceQueue(device_, indices.graphics_family.value(), 0,
                   &graphics_queue_);
  vkGetDeviceQueue(device_, indices.presentation_family.value(), 0,
                   &presentation_queue_);
}

VulkanEngine::QueueFamilyIndices
VulkanEngine::FindQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices;
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                           queue_families.data());
  int i = 0;
  for (const auto &queue_family : queue_families) {
    if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphics_family = i;
    }
    VkBool32 presentation_support = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_,
                                         &presentation_support);
    if (presentation_support) {
      indices.presentation_family = i;
    }
    if (indices.IsComplete()) {
      break;
    }
    ++i;
  }
  return indices;
}

void VulkanEngine::CreateSwapChain() {
  SwapChainSupportDetails swap_chain_support =
      QuerySwapChainSupport(physical_device_);

  VkSurfaceFormatKHR surface_format =
      ChooseSwapSurfaceFormat(swap_chain_support.formats);
  VkPresentModeKHR present_mode =
      ChooseSwapPresentMode(swap_chain_support.present_modes);
  VkExtent2D extent = ChooseSwapExtent(swap_chain_support.capabilities);

  uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
  if (swap_chain_support.capabilities.maxImageCount > 0 &&
      image_count > swap_chain_support.capabilities.maxImageCount) {
    image_count = swap_chain_support.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface_;
  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format.format;
  create_info.imageColorSpace = surface_format.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  QueueFamilyIndices indices = FindQueueFamilies(physical_device_);
  uint32_t queue_family_indices[] = {indices.graphics_family.value(),
                                     indices.presentation_family.value()};
  if (indices.graphics_family != indices.presentation_family) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
  }
  create_info.preTransform = swap_chain_support.capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  create_info.presentMode = present_mode;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) !=
      VK_SUCCESS) {
    spdlog::error("Failed to create swap chain.");
    return;
  }

  vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
  swap_chain_images_.resize(image_count);
  vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count,
                          swap_chain_images_.data());

  swap_chain_image_format_ = surface_format.format;
  swap_chain_extent_ = extent;
}

VkSurfaceFormatKHR VulkanEngine::ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &available_formats) {
  for (const auto &available_format : available_formats) {
    if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return available_format;
    }
  }
  return available_formats[0];
}

VkPresentModeKHR VulkanEngine::ChooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &available_present_modes) {
  for (const auto &available_present_mode : available_present_modes) {
    if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return available_present_mode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D
VulkanEngine::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }

  int width, height;
  SDL_GetWindowSizeInPixels(window_, &width, &height);
  VkExtent2D actual_extent = {static_cast<uint32_t>(width),
                              static_cast<uint32_t>(height)};
  actual_extent.width =
      std::clamp(actual_extent.width, capabilities.minImageExtent.width,
                 capabilities.maxImageExtent.width);
  actual_extent.height =
      std::clamp(actual_extent.height, capabilities.minImageExtent.height,
                 capabilities.maxImageExtent.height);

  return actual_extent;
}

VulkanEngine::SwapChainSupportDetails
VulkanEngine::QuerySwapChainSupport(VkPhysicalDevice device) {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_,
                                            &details.capabilities);
  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count,
                                       nullptr);
  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count,
                                         details.formats.data());
  }

  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_,
                                            &present_mode_count, nullptr);
  if (present_mode_count != 0) {
    details.present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_,
                                              &present_mode_count, nullptr);
  }

  return details;
}

void VulkanEngine::CreateImageViews() {
  swap_chain_image_views_.resize(swap_chain_images_.size());
  for (size_t i = 0; i < swap_chain_images_.size(); ++i) {
    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = swap_chain_images_[i];
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = swap_chain_image_format_;
    create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &create_info, nullptr,
                          &swap_chain_image_views_[i]) != VK_SUCCESS) {
      spdlog::error("Failed to create image views.");
      return;
    }
  }
}
