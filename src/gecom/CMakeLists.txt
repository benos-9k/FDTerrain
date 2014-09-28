

SET(GECOM_SOURCES 
	Log.cpp
	Concurrent.cpp
	Window.cpp
	Chrono.cpp
	
)

SET(GECOM_HEADERS
	GECom.hpp
	GL.hpp
	Initial3D.hpp
	Log.hpp
	Concurrent.hpp
	Window.hpp
	Chrono.hpp
	Shader.hpp
	
)

add_library(gecom ${GECOM_SOURCES} ${GECOM_HEADERS})

# list libs that must be linked with gecom
# cryptopp ${CRYPTOPP_LIBRARIES}
list(APPEND GECOM_LIBRARIES_impl glew ${GLEW_LIBRARIES} glfw ${GLFW_LIBRARIES})
if(WIN32)
	# netcode, and cryptopp depends on wsock
	#list(APPEND GECOM_LIBRARIES_impl wsock32 ws2_32)
endif()

# list macros that must be defined when using gecom
# ${CRYPTOPP_DEFINITIONS}
# ${GLFW_DEFINITIONS} - i dont think this exists
list(APPEND GECOM_DEFINITIONS_impl ${GLEW_DEFINITIONS})

# export
set(GECOM_LIBRARIES ${GECOM_LIBRARIES_impl} CACHE STRING "GECom required libraries" FORCE)
set(GECOM_DEFINITIONS ${GECOM_DEFINITIONS_impl} CACHE STRING "GECom required definitions" FORCE)

add_definitions(${GECOM_DEFINITIONS})

set_target_properties(gecom PROPERTIES LINKER_LANGUAGE CXX)
