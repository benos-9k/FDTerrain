/*
 * GECom Main Header
 *
 * aka shit that needs to go somewhere
 */

#ifndef GECOM_HPP
#define GECOM_HPP

#include <cctype>
#include <string>
#include <algorithm>
#include <memory>
#include <utility>

#include "Initial3D.hpp"

// this alias will be available by default in new i3d
namespace i3d = initial3d;

namespace gecom {
	
	class Uncopyable {
	private:
		Uncopyable(const Uncopyable &rhs) = delete;
		Uncopyable & operator=(const Uncopyable &rhs) = delete;
	protected:
		Uncopyable() { }
	};

	// real std::make_unique is c++14, so this will do for the moment
	template <typename T, typename... ArgTR>
	inline std::unique_ptr<T> make_unique(ArgTR && ...args) {
		return std::unique_ptr<T>(new T(std::forward<ArgTR>(args)...));
	}
	
	// trim leading and trailing whitespace
	inline std::string trim(const std::string &s) {
		auto wsfront = std::find_if_not(s.begin(), s.end(), [](int c) { return std::isspace(c); });
		auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) { return std::isspace(c); }).base();
		return wsback <= wsfront ? std::string() : std::string(wsfront, wsback);
	}

	// function to declare things as unused
	template <typename T1, typename... TR>
	inline void unused(const T1 &t1, const TR &...tr) {
		(void) t1;
		unused(tr...);
	}

	// unused() base case
	inline void unused() { }
}

#endif // GECOM_HPP
