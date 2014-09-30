/*
 * GECom Shader Manager
 *
 * Extra preprocessor directives:
 *		#include "..."		: include file relative to directory containing current file
 *		#include <...>		: include file relative to directories known to the shader manager
 *								#include resolves #version directives
 *
 *		#shader shader-type	: specify shader type(s) a file should be compiled as
 *								valid values for shader-type are:
 *								- vertex
 *								- fragment
 *								- geometry
 *								- tess_control
 *								- tess_evaluation
 *
 * These extra directives are processed regardless of #if etc.
 *
 * The line numbers reported in compiler messages should be correct provided the compiler
 * follows the GLSL spec for the version in question regarding the #line directive.
 * The spec changed regarding this with GLSL 330 (to the best of my knowledge).
 * If the compiler uses the pre-330 behaviour for 330 or later code, line numbers will
 * be reported as 1 greater than they should be.
 *
 * @author Ben Allen
 *
 * TODO suppose I link multiple frag shaders together, how do i tell them apart in program info log?
 * TODO better unload functions?
 * TODO #include, #shader and #version are processed regardless of #if etc - probably wontfix
 * TODO this code has gradually 'evolved' to the point that it's not very clean anymore
 * TODO shader binaries with extension GL_ARB_get_program_binary ?
 *
 */

//
// If (eg with an AMD GPU) shader compilation fails regarding #line
// #define GECOM_SHADER_NO_LINE_DIRECTIVES
// before including this file to prevent #line directives.
// This will mean line numbers will not be correct.
//

#ifndef GECOM_SHADER_HPP
#define GECOM_SHADER_HPP

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <mutex>
#include <type_traits>
#include <limits>

#include "GECom.hpp"
#include "GL.hpp"
#include "Log.hpp"

namespace gecom {

	class ShaderManager;

	class shader_error : public std::runtime_error {
	public:
		explicit inline shader_error(const std::string &what_ = "Generic shader error.") : std::runtime_error(what_) { }
	};

	class shader_type_error : public shader_error {
	public:
		explicit inline shader_type_error(const std::string &what_ = "Bad shader type.") : shader_error(what_) { }
	};

	class shader_compile_error : public shader_error {
	public:
		explicit inline shader_compile_error(const std::string &what_ = "Shader compilation failed.") : shader_error(what_) { }
	};

	class shader_link_error : public shader_error {
	public:
		explicit inline shader_link_error(const std::string &what_ = "Shader program linking failed.") : shader_error(what_) { }
	};

	inline void printShaderInfoLog(GLuint obj, bool error = false) {
		int infologLength = 0;
		int charsWritten = 0;
		glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
		if (infologLength > 1) {
			std::vector<char> infoLog(infologLength);
			glGetShaderInfoLog(obj, infologLength, &charsWritten, &infoLog[0]);
			if (error) {
				log("ShaderMan").error() << &infoLog[0];
			} else {
				// TODO might not be warnings
				log("ShaderMan").warning() << &infoLog[0];
			}
		}
	}

	inline void printProgramInfoLog(GLuint obj, bool error = false) {
		int infologLength = 0;
		int charsWritten  = 0;
		glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &infologLength);
		if (infologLength > 1) {
			std::vector<char> infoLog(infologLength);
			glGetProgramInfoLog(obj, infologLength, &charsWritten, &infoLog[0]);
			if (error) {
				log("ShaderMan").error() << &infoLog[0];
			} else {
				// TODO might not be warnings
				log("ShaderMan").warning() << &infoLog[0];
			}
		}
	}

	inline GLuint compileShader(GLenum type, const std::string &text) {
		GLuint shader = glCreateShader(type);
		const char *text_c = text.c_str();
		glShaderSource(shader, 1, &text_c, nullptr);
		try {
			glCompileShader(shader);
		} catch (gl_error &) { }
		GLint compile_status;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
		if (!compile_status) {
			printShaderInfoLog(shader, true);
			throw shader_compile_error();
		}
		// always print, so we can see warnings
		printShaderInfoLog(shader, false);
		return shader;
	}

	inline GLuint compileShader(GLenum type, std::istream &text_is) {
		std::string text;
		while (text_is.good()) {
			std::string line;
			std::getline(text_is, line);
			(text += line) += '\n';
		}
		return compileShader(type, text);
	}

	inline GLuint compileShaderFromFile(GLenum type, const std::string &path) {
		std::ifstream ifs(path.c_str());
		if (!ifs.good()) {
			std::string msg = "Error opening shader source '";
			msg += path;
			msg += "'.";
			throw shader_error(msg);
		}
		log("ShaderMan") << "Compiling '" << path << "'...";
		return compileShader(type, ifs);
	}

	inline void linkShaderProgram(GLuint prog) {
		try {
			glLinkProgram(prog);
		} catch (gl_error &) { }
		GLint link_status;
		glGetProgramiv(prog, GL_LINK_STATUS, &link_status);
		if (!link_status) {
			printProgramInfoLog(prog, true);
			throw shader_link_error();
		}
		// always print, so we can see warnings
		printProgramInfoLog(prog, false);
	}

	class shader_profile {
	public:
		unsigned version;
		std::string name;
		
		inline explicit shader_profile(unsigned version_, std::string name_ = "") : version(version_), name(name_) { }
		
		inline bool operator==(const shader_profile &rhs) const {
			return version == rhs.version && name == rhs.name;
		}
		
		inline bool operator!=(const shader_profile &rhs) const {
			return !(*this == rhs);
		}
		
		inline friend std::ostream & operator<<(std::ostream &out, const shader_profile &profile) {
			out << profile.version;
			if (!profile.name.empty()) out << " " << profile.name;
			return out;
		}
	};

	class shader_program_spec {
		friend class ShaderManager;
	private:
		std::set<std::string> m_sources;
		std::map<std::string, std::string> m_definitions;
		GLenum m_xfb_mode = 0;
		std::vector<std::string> m_xfb_varyings;

		// cached information if this spec has already been compiled
		const ShaderManager *m_shaderman = nullptr;
		std::chrono::steady_clock::time_point m_timestamp;
		GLuint m_prog;

		template <typename T0, typename... TR>
		inline void add_xfb_varyings(const T0 &v0, const TR &...vr) {
			std::ostringstream oss;
			oss << v0;
			m_xfb_varyings.push_back(oss.str());
			add_xfb_varyings(vr...);
		}

		inline void add_xfb_varyings() { }
		
	public:
		inline shader_program_spec & source(const std::string &name) {
			// trim leading / trailing whitespace, internal is allowed
			std::string name2 = trim(name);
			if (!name2.empty()) {
				m_sources.insert(name2);
			}
			return *this;
		}

		inline shader_program_spec & define(const std::string &symbol, const std::string &value) {
			// whitespace is not allowed in the symbol token
			std::istringstream iss(symbol);
			std::string symbol2;
			iss >> symbol2;
			if (!iss.fail()) {
				m_definitions[symbol2] = value;
			}
			return *this;
		}

		inline shader_program_spec & define(const std::string &symbol, uint_least32_t value) {
			std::ostringstream oss;
			oss << value << "u";
			return define(symbol, oss.str());
		}

		inline shader_program_spec & define(const std::string &symbol, int_least32_t value) {
			std::ostringstream oss;
			oss << value;
			return define(symbol, oss.str());
		}

		inline shader_program_spec & define(const std::string &symbol, bool value) {
			std::ostringstream oss;
			oss << std::boolalpha << value;
			return define(symbol, oss.str());
		}

		inline shader_program_spec & define(const std::string &symbol, float value) {
			std::ostringstream oss;
			oss << std::showpoint << std::setprecision(std::numeric_limits<float>::digits10 + 2) << value;
			return define(symbol, oss.str());
		}

		inline shader_program_spec & define(const std::string &symbol, double value) {
			std::ostringstream oss;
			oss << std::showpoint << std::setprecision(std::numeric_limits<double>::digits10 + 2) << value << "lf";
			return define(symbol, oss.str());
		}

		inline shader_program_spec & define(const std::string &symbol) {
			return define(symbol, "");
		}

		// setup input to glTransformFeedbackVaryings() using GL_INTERLEAVED_ATTRIBS
		template <typename... TR>
		inline shader_program_spec & feedbackInterleaved(const TR &...varyings) {
			m_xfb_mode = GL_INTERLEAVED_ATTRIBS;
			m_xfb_varyings.clear();
			add_xfb_varyings(varyings...);
			return *this;
		}

		// setup input to glTransformFeedbackVaryings() using GL_SEPARATE_ATTRIBS
		template <typename... TR>
		inline shader_program_spec & feedbackSeparate(const TR &...varyings) {
			m_xfb_mode = GL_SEPARATE_ATTRIBS;
			m_xfb_varyings.clear();
			add_xfb_varyings(varyings...);
			return *this;
		}

		inline const std::set<std::string> & sources() const {
			return m_sources;
		}

		inline const std::map<std::string, std::string> & definitions() const {
			return m_definitions;
		}

		inline GLenum feedbackMode() const {
			return m_xfb_mode;
		}

		inline const std::vector<std::string> feedbackVaryings() const {
			return m_xfb_varyings;
		}

		inline bool operator==(const shader_program_spec &other) const {
			return m_sources == other.m_sources && m_definitions == other.m_definitions &&
				m_xfb_mode == other.m_xfb_mode && m_xfb_varyings == other.m_xfb_varyings;
		}

		inline bool operator!=(const shader_program_spec &other) const {
			return !(*this == other);
		}

		inline friend std::ostream & operator<<(std::ostream &out, const shader_program_spec &spec) {
			std::ostringstream oss;
			for (auto s : spec.sources()) {
				oss << s << " ";
			}
			for (auto def : spec.definitions()) {
				oss << "-D" << def.first;
				if (!def.second.empty()) {
					oss << "=" << def.second;
				}
				oss << " ";
			}
			if (spec.m_xfb_mode) {
				switch (spec.m_xfb_mode) {
				case GL_INTERLEAVED_ATTRIBS:
					oss << "GL_INTERLEAVED_ATTRIBS ";
					break;
				case GL_SEPARATE_ATTRIBS:
					oss << "GL_SEPARATE_ATTRIBS ";
					break;
				default:
					oss << "???";
					// TODO
				}
			}
			out << trim(oss.str());
			return out;
		}

	};

	class ShaderManager : private Uncopyable {
	private:
		// shader cache entry
		struct shader_t {
			GLenum type;
			std::string name;
			std::map<std::string, std::string> defs;
			GLuint id;
		};

		// program cache entry
		struct program_t {
			shader_program_spec spec;
			GLuint id;
		};

		// thread-safety
		std::recursive_mutex m_mutex;

		// timestamp to determine if external cached ids are valid
		// set by ctor, reset by unload functions
		std::chrono::steady_clock::time_point m_timestamp;

		// TODO use more intelligent data structures (maps)
		// that might be hard since it involves using a map as a key...
		std::vector<std::string> m_shader_dirs;
		std::vector<shader_t> m_shaders;
		std::vector<program_t> m_programs;

		inline static std::string shaderTypeString(GLenum type) {
			switch (type) {
			case GL_VERTEX_SHADER:
				return "GL_VERTEX_SHADER";
			case GL_FRAGMENT_SHADER:
				return "GL_FRAGMENT_SHADER";
			case GL_GEOMETRY_SHADER:
				return "GL_GEOMETRY_SHADER";
			case GL_TESS_CONTROL_SHADER:
				return "GL_TESS_CONTROL_SHADER";
			case GL_TESS_EVALUATION_SHADER:
				return "GL_TESS_EVALUATION_SHADER";
			default:
				return "UNKNOWN";
			}
		}
		
		// make a line directive string, using pre-330 behaviour
		inline static std::string lineDirective(const shader_profile &profile, unsigned line, unsigned source) {
			// to set next line to 1
			// by glsl-spec-1.30.8: #line 0
			// by glsl-spec-4.20.8: #line 1
			// changeover seems to be at version 330
			std::ostringstream oss;
#ifndef GECOM_SHADER_NO_LINE_DIRECTIVES
			if (profile.version < 330) {
				oss << "#line " << line << " " << source;
			} else {
				oss << "#line " << (line + 1) << " " << source;
			}
#endif
			return oss.str();
		}

	public:
		// assumed to always use '/' as separator
		inline void addSourceDirectory(const std::string &dir) {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);
			// FIXME more robust
			if (dir[dir.length() - 1] == '/') {
				m_shader_dirs.push_back(dir);
			} else {
				m_shader_dirs.push_back(dir + '/');
			}
		}

		inline std::string resolveSourcePath(const std::string &name) {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);
			// find file
			for (auto it = m_shader_dirs.cbegin(); it != m_shader_dirs.cend(); it++) {
				std::string path = *it + name;
				std::ifstream ifs(path);
				if (ifs.good()) {
					// found it
					return path;
				}
			}
			// didnt find the file
			std::string msg = "Unable to find shader file '";
			msg += name;
			msg += "'.";
			throw shader_error(msg);
		}

		// process #include and #shader directives, and strip #version directives
		// returns the version number from this file
		// "" style includes are handled relative to the directory containing the shader being compiled
		// <> style includes are handled relative to source directories known to the ShaderManager
		inline shader_profile preprocessShader(
			const std::string &path,
			std::ostream &text_os,
			std::vector<std::string> &source_names,
			std::vector<GLenum> &shader_types,
			std::ostream &log_os
		) {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);

			char cbuf[1024];
			shader_profile profile(110);
			bool have_ver = false;
			int source_id = source_names.size();
			source_names.push_back(path);
			int line_number = 1;

			// open file
			std::ifstream ifs(path);
			if (!ifs.good()) {
				std::string msg;
				msg += "Error opening file '";
				msg += path;
				msg += "'.";
				throw shader_error(msg);
			}

			// get cwd
			std::string cwd;
			size_t index = path.find_last_of('/');
			if (index != std::string::npos) {
				cwd = path.substr(0, index + 1);
			} else {
				// assume actual current directory
				cwd = "./";
			}

			// don't init line numbers till we have a #version

			while (ifs.good()) {
				// TODO like this, borked #includes will be passed to the compiler
				std::string line;
				std::getline(ifs, line);
				int v;

				if (std::sscanf(line.c_str(), " #version %d %s", &v, cbuf) == 1 && !have_ver) {
					// deal with #version version-id
					// this strips all #version directives, and only remembers the first
					profile.version = v;
					profile.name = "";
					have_ver = true;
					// init line numbers
					text_os << lineDirective(profile, line_number, source_id) << '\n';

				} else if (std::sscanf(line.c_str(), " #version %d %s", &v, cbuf) == 2 && !have_ver) {
					// deal with #version version-id profile-name
					// this strips all #version directives, and only remembers the first
					profile.version = v;
					profile.name = cbuf;
					have_ver = true;
					// init line numbers
					text_os << lineDirective(profile, line_number, source_id) << '\n';

				} else if (std::sscanf(line.c_str(), " #include \"%[^\"]\"", cbuf) > 0) {
					// deal with #include "..."
					// the negated charset is C99
					std::string path_inc = cwd + trim(cbuf);
					shader_profile profile_inc = preprocessShader(path_inc, text_os, source_names, shader_types, log_os);
					if (profile != profile_inc) {
						log_os << "WARNING: \n'" << path_inc << "' (" << profile_inc
							<< ") included by\n'" << path << "' (" << profile << ")" << std::endl;
					}
					text_os << lineDirective(profile, line_number, source_id) << '\n';

				} else if (std::sscanf(line.c_str(), " #include <%[^>]>", cbuf) > 0) {
					// deal with #include <...>
					// the negated charset is C99
					std::string path_inc = resolveSourcePath(cbuf);
					shader_profile profile_inc = preprocessShader(path_inc, text_os, source_names, shader_types, log_os);
					if (profile != profile_inc) {
						log_os << "WARNING: \n'" << path_inc << "' (" << profile_inc
							<< ") included by\n'" << path << "' (" << profile << ")" << std::endl;
					}
					text_os << lineDirective(profile, line_number, source_id) << '\n';

				} else if (std::sscanf(line.c_str(), " #shader %s", cbuf) > 0) {
					// deal with #shader - specifies shader type to compile as
					// TODO this silently ignores bad #shader directives
					std::string type_str = cbuf;
					if (type_str == "vertex") shader_types.push_back(GL_VERTEX_SHADER);
					if (type_str == "fragment") shader_types.push_back(GL_FRAGMENT_SHADER);
					if (type_str == "geometry") shader_types.push_back(GL_GEOMETRY_SHADER);
					if (type_str == "tess_control") shader_types.push_back(GL_TESS_CONTROL_SHADER);
					if (type_str == "tess_evaluation") shader_types.push_back(GL_TESS_EVALUATION_SHADER);
					text_os << '\n';
					
				} else {
					// ... its a normal glsl line
					text_os << line << '\n';
				}

				line_number++;
			}
			return profile;
		}

		// compile shader or return cached shader id
		inline GLuint shader(GLenum type, const std::string &name, const std::map<std::string, std::string> &definitions, bool force_type = false) {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);

			std::string name_stripped = trim(name);

			// is it already compiled?
			for (auto it = m_shaders.cbegin(); it != m_shaders.cend(); it++) {
				if (it->type == type && it->name == name_stripped && it->defs == definitions) {
					// yes, return id
					return it->id;
				}
			}

			// resolve path
			std::string path = resolveSourcePath(name_stripped);

			// preprocess
			std::ostringstream text_os_main;
			std::vector<std::string> source_names;
			std::vector<GLenum> shader_types;
			std::ostringstream log_os;
			shader_profile profile = preprocessShader(path, text_os_main, source_names, shader_types, log_os);
			
			if (!force_type && std::find(shader_types.cbegin(), shader_types.cend(), type) == shader_types.cend()) {
				// type being compiled was not declared (and the type isnt being forced)
				throw shader_type_error("Shader type being compiled was not declared.");
			}
			
			{
				auto compile_log = log("ShaderMan");

				if (!log_os.str().empty()) {
					// pre-pre-processor log isnt empty - warning
					compile_log.warning();
				}

				compile_log << "Compiling " << shaderTypeString(type) << " (" << profile << ") '" << path << "'..." << std::endl;

				// display source string info
				for (size_t i = 0; i < source_names.size(); i++) {
					compile_log << i << "\t: " << source_names[i] << std::endl;
				}
			
				// display any output from pre-pre-processor
				compile_log << log_os.str();
			}
			
			// specify version / profile and define type
			std::ostringstream text_os;
			text_os << "#version " << profile << '\n';
			switch(type) {
			case GL_VERTEX_SHADER:
				text_os << "#define _VERTEX_\n";
				break;
			case GL_FRAGMENT_SHADER:
				text_os << "#define _FRAGMENT_\n";
				break;
			case GL_GEOMETRY_SHADER:
				text_os << "#define _GEOMETRY_\n";
				break;
			case GL_TESS_CONTROL_SHADER:
				text_os << "#define _TESS_CONTROL_\n";
				break;
			case GL_TESS_EVALUATION_SHADER:
				text_os << "#define _TESS_EVALUATION_\n";
				break;
			default:
				throw shader_error("Unknown shader type.");
			}

			// append supplied definitions
			for (auto it = definitions.cbegin(); it != definitions.cend(); it++) {
				text_os << "#define " << it->first << " " << it->second << "\n";
			}

			// append preprocessed source
			text_os << text_os_main.str();
			
			// now, lets compile!
			GLuint id = compileShader(type, text_os.str());
			
			// add to cache
			shader_t shader;
			shader.type = type;
			shader.name = name_stripped;
			shader.defs = definitions;
			shader.id = id;
			m_shaders.push_back(shader);
			return id;
		}

		// compile shader program or return cached program id
		inline GLuint program(const shader_program_spec &spec) {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);

			// is it cached in the spec object?
			if (spec.m_shaderman == this && spec.m_timestamp > m_timestamp) {
				// yes, and its valid (newer than last external cache invalidation)
				return spec.m_prog;
			}

			// is it already linked?
			for (auto it = m_programs.cbegin(); it != m_programs.cend(); it++) {
				if (it->spec == spec) {
					// yes, return id
					return it->id;
				}
			}

			// nope, compile as necessary then link
			GLuint id = glCreateProgram();
			for (auto it = spec.sources().cbegin(); it != spec.sources().cend(); it++) {
				size_t i = it->find_last_of(".");
				std::string ext = it->substr(i);
				// if the ext is for a specific type, only compile as that type
				if (ext == ".vert") {
					glAttachShader(id, shader(GL_VERTEX_SHADER, *it, spec.definitions(), true));
				} else if (ext == ".frag") {
					glAttachShader(id, shader(GL_FRAGMENT_SHADER, *it, spec.definitions(), true));
				} else if (ext == ".geom") {
					glAttachShader(id, shader(GL_GEOMETRY_SHADER, *it, spec.definitions(), true));
				} else {
					// unable to determine type from extension, try everything
					static const std::vector<GLenum> types {
						GL_VERTEX_SHADER,
						GL_FRAGMENT_SHADER,
						GL_GEOMETRY_SHADER,
						GL_TESS_CONTROL_SHADER,
						GL_TESS_EVALUATION_SHADER
					};
					for (GLenum type : types) {
						try {
							glAttachShader(id, shader(type, *it, spec.definitions(), false));
						} catch (shader_type_error &e) {
							// type wasnt declared in source, skip
						}
					}
				}
			}

			log("ShaderMan") << "Linking shader program '" << spec << "'...";
			linkShaderProgram(id);
			log("ShaderMan") << "Shader program compiled and linked successfully.";

			// cache it
			program_t program;
			program.spec = spec;
			program.id = id;
			m_programs.push_back(program);
			return id;
		}

		// compile shader program or return cached program id, caching id in the supplied spec
		inline GLuint program(shader_program_spec &spec) {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);
			GLuint prog = program(static_cast<const shader_program_spec &>(spec));
			spec.m_prog = prog;
			spec.m_shaderman = this;
			spec.m_timestamp = std::chrono::steady_clock::now();
			return prog;
		}

		// unload / delete all shaders and programs
		inline void unloadAll() {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);
			glUseProgram(0);
			// delete programs
			for (auto it = m_programs.cbegin(); it != m_programs.cend(); it++) {
				glDeleteProgram(it->id);
			}
			// delete shaders
			for (auto it = m_shaders.cbegin(); it != m_shaders.cend(); it++) {
				glDeleteShader(it->id);
			}
			// clear cache
			m_shaders.clear();
			m_programs.clear();
			// set timestamp
			m_timestamp = std::chrono::steady_clock::now();
		}
		
		// get a vector of all loaded program spec(ification)s
		std::vector<shader_program_spec> loadedProgramSpecs() {
			std::lock_guard<std::recursive_mutex> lock(m_mutex);
			std::vector<shader_program_spec> prog_specs;
			for (const program_t & prog : m_programs) {
				prog_specs.push_back(prog.spec);
			}
			return prog_specs;
		}

		// ctor takes a source directory
		inline ShaderManager(const std::string &dir) {
			addSourceDirectory(dir);
			m_timestamp = std::chrono::steady_clock::now();
		}
	};

}

#endif // GECOM_SHADER_HPP
