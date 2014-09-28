#include <cassert>
#include <cstdlib>
#include <map>
#include <bitset>
#include <sstream>

#include "Window.hpp"

using namespace std;

namespace gecom {

	namespace {

		struct WindowData {
			Window *window;
			GLEWContext context;
			bool glew_init_done = false;
			std::bitset<GLFW_KEY_LAST + 1> vk;
			std::bitset<GLFW_MOUSE_BUTTON_LAST + 1> mb;
			WindowData(Window *window_) : window(window_) { }
		};

		WindowData * getWindowData(GLFWwindow *handle) {
			return (WindowData *) glfwGetWindowUserPointer(handle);
		}

		Window * getWindow(GLFWwindow *handle) {
			return getWindowData(handle)->window;
		}

		void callbackWindowPos(GLFWwindow *handle, int x, int y) {
			Window *win = getWindow(handle);
			window_pos_event e;
			e.window = win;
			e.pos = point2i(x, y);
			win->onMove.notify(e);
		}

		void callbackWindowSize(GLFWwindow *handle, int w, int h) {
			Window *win = getWindow(handle);
			window_size_event e;
			e.window = win;
			e.size = size2i(w, h);
			win->onResize.notify(e);
		}

		void callbackWindowClose(GLFWwindow *handle) {
			Window *win = getWindow(handle);
			window_event e;
			e.window = win;
			win->onClose.notify(e);
		}

		void callbackWindowRefresh(GLFWwindow *handle) {
			Window *win = getWindow(handle);
			window_event e;
			e.window = win;
			win->onRefresh.notify(e);
		}

		void callbackWindowFocus(GLFWwindow *handle, int focused) {
			WindowData *wd = getWindowData(handle);
			if (!focused) {
				// lost focus, release all keys and buttons
				wd->vk.reset();
				wd->mb.reset();
			}
			window_focus_event e;
			e.window = wd->window;
			e.focused = focused;
			wd->window->onFocus.notify(e);
			if (focused) {
				wd->window->onFocusGain.notify(e);
			} else {
				wd->window->onFocusLose.notify(e);
			}
		}

		void callbackWindowIconify(GLFWwindow *handle, int iconified) {
			Window *win = getWindow(handle);
			window_icon_event e;
			e.window = win;
			e.iconified = iconified;
			win->onIcon.notify(e);
			if (iconified) {
				win->onMinimise.notify(e);
			} else {
				win->onRestore.notify(e);
			}
		}

		void callbackFramebufferSize(GLFWwindow *handle, int w, int h) {
			// TODO
		}

		void callbackMouseButton(GLFWwindow *handle, int button, int action, int mods) {
			// i dont think mouse buttons get repeats, but whatever
			WindowData *wd = getWindowData(handle);
			if (action == GLFW_PRESS || action == GLFW_REPEAT) {
				wd->vk.set(button, true);
			} else {
				wd->vk.set(button, false);
			}
			mouse_button_event e;
			e.window = wd->window;
			e.button = button;
			e.action = action;
			e.mods = mods;
			e.entered = false;
			e.exited = false;
			glfwGetCursorPos(handle, &e.pos.x, &e.pos.y);
			wd->window->onMouse.notify(e);
			if (action == GLFW_PRESS || action == GLFW_REPEAT) {
				wd->window->onMousePress.notify(e);
			} else {
				wd->window->onMouseRelease.notify(e);
			}
		}

		void callbackCursorPos(GLFWwindow *handle, double x, double y) {
			Window *win = getWindow(handle);
			mouse_event e;
			e.window = win;
			e.pos = point2d(x, y);
			e.entered = false;
			e.exited = false;
			win->onMouseMove.notify(e);
		}

		void callbackCursorEnter(GLFWwindow *handle, int entered) {
			Window *win = getWindow(handle);
			mouse_event e;
			e.window = win;
			glfwGetCursorPos(handle, &e.pos.x, &e.pos.y);
			e.entered = entered;
			e.exited = !entered;
			win->onMouseMove.notify(e);
			if (entered) {
				win->onMouseEnter.notify(e);
			} else {
				win->onMouseExit.notify(e);
			}
		}

		void callbackScroll(GLFWwindow *handle, double xoffset, double yoffset) {
			Window *win = getWindow(handle);
			mouse_scroll_event e;
			e.window = win;
			glfwGetCursorPos(handle, &e.pos.x, &e.pos.y);
			e.entered = false;
			e.exited = false;
			e.offset = size2d(xoffset, yoffset);
			win->onScroll.notify(e);
		}

		void callbackKey(GLFWwindow *handle, int key, int scancode, int action, int mods) {
			WindowData *wd = getWindowData(handle);
			if (action == GLFW_PRESS || action == GLFW_REPEAT) {
				wd->vk.set(key, true);
			} else {
				wd->vk.set(key, false);
			}
			key_event e;
			e.window = wd->window;
			e.key = key;
			e.scancode = scancode;
			e.action = action;
			e.mods = mods;
			wd->window->onKey.notify(e);
			if (action == GLFW_PRESS || action == GLFW_REPEAT) {
				wd->window->onKeyPress.notify(e);
			} else {
				wd->window->onKeyRelease.notify(e);
			}
		}

		void callbackChar(GLFWwindow *handle, unsigned codepoint) {
			Window *win = getWindow(handle);
			char_event e;
			e.window = win;
			e.codepoint = codepoint;
			win->onChar.notify(e);
		}

		void callbackErrorGLFW(int error, const char *description) {
			gecom::log("GLFW").error() << "Error " << error << " : " << description;
		}

		// init GLFW global state.
		// this should only be called from the main thread.
		// does nothing if already initialised.
		void init_glfw() {
			static bool done = false;
			if (!done) {
				gecom::log("Window") << "GLFW initialising...";
				// this is safe to call before glfwInit()
				glfwSetErrorCallback(callbackErrorGLFW);
				if (!glfwInit()) {
					gecom::log("Window").error() << "GLFW initialisation failed";
					// screw catching this, ever
					std::abort();
				}
				gecom::log("Window") % 0 << "GLFW initialised";
				done = true;
			}
		}

		// GL callback for debug information
		void APIENTRY callbackDebugGL(
			GLenum source,
			GLenum type,
			GLuint id,
			GLenum severity,
			GLsizei length,
			const GLchar *message,
			void *userParam
		){
			// enum documentation:
			// https://www.opengl.org/sdk/docs/man4/html/glDebugMessageControl.xhtml
			
			// message source within GL -> log source
			string log_source = "GL";
			switch (source) {
			case GL_DEBUG_SOURCE_API:
				log_source = "GL:API";
				break;
			case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
				log_source = "GL:Window";
				break;
			case GL_DEBUG_SOURCE_SHADER_COMPILER:
				log_source = "GL:Shader";
				break;
			case GL_DEBUG_SOURCE_THIRD_PARTY:
				log_source = "GL:ThirdParty";
				break;
			case GL_DEBUG_SOURCE_APPLICATION:
				log_source = "GL:App";
				break;
			case GL_DEBUG_SOURCE_OTHER:
				log_source = "GL:Other";
				break;
			default:
				break;
			}

			// piecewise construct log message
			auto logs = log(log_source);
			bool exceptional = false;

			// message type -> log type
			switch (type) {
			case GL_DEBUG_TYPE_ERROR:
				logs.error();
				logs << "Error";
				exceptional = true;
				break;
			case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
				logs.warning();
				logs << "Deprecated Behaviour";
				break;
			case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
				logs.warning();
				logs << "Undefined Behaviour";
				break;
			case GL_DEBUG_TYPE_PORTABILITY:
				logs.warning();
				logs << "Portability";
				break;
			case GL_DEBUG_TYPE_PERFORMANCE:
				logs.warning();
				logs << "Performance";
				break;
			case GL_DEBUG_TYPE_MARKER:
				logs << "Marker";
				break;
			case GL_DEBUG_TYPE_PUSH_GROUP:
				logs << "Push Group";
				break;
			case GL_DEBUG_TYPE_POP_GROUP:
				logs << "Pop Group";
				break;
			case GL_DEBUG_TYPE_OTHER:
				logs << "Other";
				break;
			}

			// severity -> log verbosity
			switch (severity) {
			case GL_DEBUG_SEVERITY_NOTIFICATION:
				logs % 3;
				break;
			case GL_DEBUG_SEVERITY_LOW:
				logs % 2;
				break;
			case GL_DEBUG_SEVERITY_MEDIUM:
				logs % 1;
				break;
			case GL_DEBUG_SEVERITY_HIGH:
				logs % 0;
				break;
			}

			// actual message. id = as returned by glGetError()
			ostringstream oss;
			oss << " [" << id << "] : " << message;
			logs << oss.str();
			
#ifndef GECOM_GL_NO_EXCEPTIONS
			if (exceptional) {
				throw gl_error(); //oss.str());
			}
#endif

		}

	}

	void * getGlewContext() {
		GLFWwindow *handle = glfwGetCurrentContext();
		if (!handle) {
			throw window_error("no current context");
		}
		return &(getWindowData(handle)->context);
	}

	void Window::initialise() {
		// set ALL the callbacks
		glfwSetWindowPosCallback(m_handle, callbackWindowPos);
		glfwSetWindowSizeCallback(m_handle, callbackWindowSize);
		glfwSetWindowCloseCallback(m_handle, callbackWindowClose);
		glfwSetWindowRefreshCallback(m_handle, callbackWindowRefresh);
		glfwSetWindowFocusCallback(m_handle, callbackWindowFocus);
		glfwSetWindowIconifyCallback(m_handle, callbackWindowIconify);
		glfwSetFramebufferSizeCallback(m_handle, callbackFramebufferSize);
		glfwSetMouseButtonCallback(m_handle, callbackMouseButton);
		glfwSetCursorPosCallback(m_handle, callbackCursorPos);
		glfwSetCursorEnterCallback(m_handle, callbackCursorEnter);
		glfwSetScrollCallback(m_handle, callbackScroll);
		glfwSetKeyCallback(m_handle, callbackKey);
		glfwSetCharCallback(m_handle, callbackChar);
		// create a windowdata object
		glfwSetWindowUserPointer(m_handle, new WindowData(this));
	}

	void Window::destroy() {
		// must only be called from the main thread
		delete getWindowData(m_handle);
		glfwDestroyWindow(m_handle);
	}

	void Window::makeContextCurrent() {
		glfwMakeContextCurrent(m_handle);
		WindowData *wd = getWindowData(m_handle);
		// init glew
		if (!wd->glew_init_done) {
			gecom::log("Window") << "GLEW initialising...";
			glewExperimental = true;
			GLenum glew_err = glewInit();
			//GLenum t_err = glGetError();
			//gecom::log("Window") << "GLEW t_err: " << t_err;
			gecom::log("Window") << "GLEW initialisation returned " << glew_err;
			if (glew_err != GLEW_OK) {
				gecom::log("Window").error() << "GLEW initialisation failed: " << glewGetErrorString(glew_err);
				glfwTerminate();
				std::abort();
			}
			// clear any GL errors from glew init
			GLenum gl_err = glGetError();
			while (gl_err) {
				gecom::log("Window") << "GLEW initialistion left GL error " << gl_err;
				gl_err = glGetError();
			}
			//gecom::log("Window") << "GL Error: " << gluErrorString(gl_err);
			gecom::log("Window") << "GL version string: " << glGetString(GL_VERSION);
			gecom::log("Window") % 0 << "GLEW initialised";
			wd->glew_init_done = true;
			// enable GL_ARB_debug_output if available
			if (glfwExtensionSupported("GL_ARB_debug_output")) {
				// this allows the error location to be determined from a stacktrace
				glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
				// set the callback
				glDebugMessageCallback(callbackDebugGL, this);
				glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, true);
				log("Window") << "GL debug callback installed";
			} else {
				log("Window") << "GL_ARB_debug_output not available";
			}
		}
	}

	bool Window::getKey(int key) {
		return getWindowData(m_handle)->vk.test(key);
	}

	bool Window::pollKey(int key) {
		WindowData *wd = getWindowData(m_handle);
		bool b = wd->vk.test(key);
		wd->vk.reset(key);
		return b;
	}

	bool Window::getMouseButton(int button) {
		return getWindowData(m_handle)->mb.test(button);
	}

	bool Window::pollMouseButton(int button) {
		WindowData *wd = getWindowData(m_handle);
		bool b = wd->mb.test(button);
		wd->mb.reset(button);
		return b;
	}

	Window * Window::currentContext() {
		GLFWwindow *handle = glfwGetCurrentContext();
		if (handle == nullptr) return nullptr;
		return getWindow(handle);
	}

	// this should only be called from the main thread
	create_window_args::operator Window * () {
		gecom::log("Window") % 0 << "GLFW creating window... [title=" << m_title << "]";
		init_glfw();
		glfwDefaultWindowHints();
		//GLenum gl_err = glGetError();
		//gecom::log("Window") % 0 << "GLerror: " << gl_err;
		// the context version hints are always set
		int cv = m_hints[GLFW_CONTEXT_VERSION_MAJOR] * 10 + m_hints[GLFW_CONTEXT_VERSION_MINOR];
		for (auto me : m_hints) {
			// prevent GLFW from issuing errors about (some) 3.x things not existing when asking for older version
			if (me.first == GLFW_OPENGL_PROFILE && cv < 32) continue;
			if (me.first == GLFW_OPENGL_FORWARD_COMPAT && cv < 30) continue;
			glfwWindowHint(me.first, me.second);
		}
		GLFWwindow *handle = glfwCreateWindow(m_size.w, m_size.h, m_title.c_str(), m_monitor, m_share ? m_share->handle() : nullptr);
		glfwDefaultWindowHints();
		if (!handle) {
			gecom::log("Window").error() << "GLFW window creation failed";
			throw window_error("GLFW window creation failed");
		}
		gecom::log("Window") % 0 << "GLFW window created";
		return new Window(handle, m_share);
	}


}
