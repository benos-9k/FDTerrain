
#version 330 core

#shader vertex
#shader geometry
#shader fragment

uniform sampler2D sampler_hmap;

#ifdef _VERTEX_

out flat int id;

void main() {
	id = gl_InstanceID;
}

#endif

#ifdef _GEOMETRY_

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in flat int id[];

void main() {
	
	ivec2 ts = textureSize(sampler_hmap, 0);

	ivec2 t = ivec2(id[0] % ts.x, id[0] / ts.x);



}

#endif

#ifdef _FRAGMENT_

void main() {
	
}

#endif
