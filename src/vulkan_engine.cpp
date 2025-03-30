#include "vulkan_engine.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <cstddef>
#include <optional>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_core.h>

namespace {

const std::vector<const char *> validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

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

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;

  bool IsComplete() { return graphics_family.has_value(); }
};

QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
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
    if (indices.IsComplete()) {
      break;
    }
    ++i;
  }
  return indices;
}

} // namespace

void VulkanEngine::Init() {
  // TODO: Check if everything initialized correctly
  InitSDL();
  InitVulkanInstance();
  PickPhysicalDevice();
  CreateLogicalDevice();
}

void VulkanEngine::Run() {
  SDL_Event event;
  while (running_) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT) {
        running_ = false;
      }
    }

    SDL_SetRenderDrawColor(fake_renderer_, 0, 0, 0, 255);
    SDL_RenderClear(fake_renderer_);
    SDL_RenderPresent(fake_renderer_);
    // Slow down to 60 FPS
    SDL_Delay(16);
  }
}

void VulkanEngine::Destroy() {
  vkDestroyDevice(device_, nullptr);

  if (kEnableValidationLayers) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance_, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
      func(instance_, debug_messenger_, nullptr);
    }
  }

  vkDestroyInstance(instance_, nullptr);
  SDL_DestroyRenderer(fake_renderer_); // remove it once Vulkan starts
  SDL_DestroyWindow(window_);
  SDL_Quit();
}

void VulkanEngine::InitSDL() {

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
  fake_renderer_ = SDL_CreateRenderer(window_, NULL);
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
  create_info.enabledLayerCount = 0;

  if (kEnableValidationLayers) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    create_info.ppEnabledLayerNames = validation_layers.data();
  } else {
    create_info.enabledLayerCount = 0;
  }

  if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
    spdlog::error("Failed to create Vulkan instance");
    return;
  }

  // ListAvailableExtensions();

  SetupDebugMessenger();
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
  for (const char *layer_name : validation_layers) {
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
  return indices.IsComplete();
}

void VulkanEngine::CreateLogicalDevice() {
  QueueFamilyIndices indices = FindQueueFamilies(physical_device_);
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = indices.graphics_family.value();
  queue_create_info.queueCount = 1;

  float queue_priority = 1.0f;
  queue_create_info.pQueuePriorities = &queue_priority;

  VkPhysicalDeviceFeatures device_features{};

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pQueueCreateInfos = &queue_create_info;
  create_info.queueCreateInfoCount = 1;
  create_info.pEnabledFeatures = &device_features;

  if (kEnableValidationLayers) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(validation_layers.size());
    create_info.ppEnabledLayerNames = validation_layers.data();
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
}
