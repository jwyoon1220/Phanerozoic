#include "engine/platform/Window.hpp"

#include "engine/core/Log.hpp"

#include <GLFW/glfw3.h>

namespace phx {

Window::Window(u32 width, u32 height, const char* title) {
    glfwSetErrorCallback([](int code, const char* msg) { log::error("GLFW error {}: {}", code, msg); });
    if (!glfwInit()) PHX_FATAL("glfwInit failed");
    if (!glfwVulkanSupported()) PHX_FATAL("GLFW reports no Vulkan loader");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(int(width), int(height), title, nullptr, nullptr);
    if (!window_) PHX_FATAL("glfwCreateWindow failed");
    glfwSetWindowUserPointer(window_, this);

    glfwSetKeyCallback(window_, [](GLFWwindow* w, int key, int, int action, int) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        if (action == GLFW_PRESS || action == GLFW_REPEAT) self->pending_.keysPressed.push_back(key);
    });
    glfwSetScrollCallback(window_, [](GLFWwindow* w, double, double dy) {
        static_cast<Window*>(glfwGetWindowUserPointer(w))->pending_.scroll += dy;
    });
    glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int, int action, int) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        self->dragging_ = action == GLFW_PRESS;
        glfwGetCursorPos(w, &self->lastX_, &self->lastY_);
    });
    glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        if (self->dragging_ || self->captured_) {
            self->pending_.mouseDx += x - self->lastX_;
            self->pending_.mouseDy += y - self->lastY_;
        }
        self->lastX_ = x;
        self->lastY_ = y;
    });
}

Window::~Window() {
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

bool Window::shouldClose() const { return glfwWindowShouldClose(window_); }
void Window::setCursorCaptured(bool on) {
    captured_ = on;
    glfwSetInputMode(window_, GLFW_CURSOR, on ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    if (on && glfwRawMouseMotionSupported()) glfwSetInputMode(window_, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    glfwGetCursorPos(window_, &lastX_, &lastY_);
}

void Window::requestClose() { glfwSetWindowShouldClose(window_, GLFW_TRUE); }
void Window::pollEvents() { glfwPollEvents(); }
void Window::waitEvents(double timeoutSeconds) { glfwWaitEventsTimeout(timeoutSeconds); }
void Window::setTitle(const std::string& title) { glfwSetWindowTitle(window_, title.c_str()); }

bool Window::keyDown(int key) const { return glfwGetKey(window_, key) == GLFW_PRESS; }

InputFrame Window::drainInput() {
    InputFrame out = std::move(pending_);
    pending_ = {};
    return out;
}

Extent2D Window::framebufferSize() const {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window_, &w, &h);
    return {u32(w > 0 ? w : 0), u32(h > 0 ? h : 0)};
}

} // namespace phx
