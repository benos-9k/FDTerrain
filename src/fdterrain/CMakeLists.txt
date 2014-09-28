

SET(FDTERRAIN_SOURCES 
	Main.cpp
	
)

SET(FDTERRAIN_HEADERS
	Float3.hpp
	
)

add_executable(fdterrain ${FDTERRAIN_SOURCES} ${FDTERRAIN_HEADERS})

add_definitions(${GECOM_DEFINITIONS})

target_link_libraries(fdterrain gecom ${GECOM_LIBRARIES})

set_target_properties(fdterrain PROPERTIES LINKER_LANGUAGE CXX)
