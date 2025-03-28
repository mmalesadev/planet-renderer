#include <SDL3/SDL.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_video.h>
#include <spdlog/spdlog.h>

int main() {
  spdlog::info("Starting Planet Renderer");
  SDL_SetAppMetadata("Planet Renderer", "0.0.1", "com.example.planet_renderer");
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    spdlog::error("Failed to initialize SDL: {}", SDL_GetError());
    return 1;
  }

  SDL_WindowFlags window_flags =
      (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE |
                        SDL_WINDOW_HIGH_PIXEL_DENSITY);
  SDL_Window *window =
      SDL_CreateWindow("PlanetRenderer", 800, 600, window_flags);

  if (!window) {
    spdlog::error("Failed to create window: {}", SDL_GetError());
    SDL_Quit();
    return 1;
  }

  int width, height, bbwidth, bbheight;
  SDL_GetWindowSize(window, &width, &height);
  SDL_GetWindowSizeInPixels(window, &bbwidth, &bbheight);
  SDL_Log("Window size: %ix%i", width, height);
  SDL_Log("Backbuffer size: %ix%i", bbwidth, bbheight);
  if (width != bbwidth) {
    SDL_Log("This is a highdpi environment.");
  }
  bool running = true;
  SDL_ShowWindow(window);

  SDL_Renderer *fake_renderer = SDL_CreateRenderer(window, NULL);
  //
  SDL_Event event;
  while (running) {
    spdlog::info("Running");
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_EVENT_QUIT) {
        running = false;
      }
    }

    SDL_SetRenderDrawColor(fake_renderer, 0, 0, 0, 255);
    SDL_RenderClear(fake_renderer);
    // Slow down to 60 FPS
    SDL_Delay(16);
  }

  SDL_RenderPresent(fake_renderer);
  SDL_DestroyRenderer(fake_renderer); // remove it once Vulkan starts
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
