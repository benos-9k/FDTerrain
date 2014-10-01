/*
 * Shader for running force-directed terrain layout with transform feedback
 */

#version 330 core

#include "nodes.glsl"

#shader vertex
#shader geometry
#shader fragment

const float timestep = 0.0001;

#ifdef FD_MOVE
uniform bool should_write_total_speed;
#endif

#ifdef _VERTEX_

flat out int id;

void main() {
	id = gl_InstanceID;
}

#endif

// run the simulation, 1 vertex (point) per node
// input comes from the textures because size limitations
// input node identified by instance id
// output is rasterized back to textures
#ifdef _GEOMETRY_

layout(points) in;
layout(points, max_vertices = 2) out;

flat in int id[];
flat out vec4 vertex_data;

void emitTexel(ivec2 tx, vec4 data) {
	vertex_data = data;
	gl_Position = vec4(nodeTexCoord(tx) * 2.0 - 1.0, 0.0, 1.0);
	EmitVertex();
	EndPrimitive();
}

void emitTexel(int id, vec4 data) {
	emitTexel(nodeTexelFromID(id), data);
}

void main() {

	Node n0 = nodeGet(id[0]);

// calc accelerations, udpate velocities
#ifdef FD_UPDATE
	
	vec2 f = vec2(0.0);

	// containment force
	vec2 fff = vec2(-n0.p) * 0.65;
	fff = fff * fff * fff;
	f += fff * 200.0 * float(num_nodes);

	// charge repulsion from every other node
	for (int i = 0; i < num_nodes; i++) {
		Node n1 = nodeGet(i);
		if (i == id[0]) continue;
		// direction is away from other node
		vec2 v = n0.p - n1.p;
		float d2 = dot(v, v);
		float k = min((1.0 / d2) * 70.0, 100000.0); // charge constant
		vec2 fc = normalize(v) * k;
		// TODO better nan handling?
		f += mix(fc, vec2(1.0), isnan(fc));
	}

	// spring contraction from connected nodes
	for (int i = 0; i < 4; i++) {
		if (n0.e[i] < 0) break;
		Node n1 = nodeGet(n0.e[i]);
		// direction is towards other node
		vec2 v = n1.p - n0.p;
		f += 1000000.0 * v; // spring constant
	}

	// acceleration
	vec2 a = f / n0.m;

	// velocity delta
	vec2 dv = a * timestep;

	if (n0.m < 1e3) {
		emitTexel(id[0], vec4(dv, 0.0, 0.0));
	}
	
#endif
	
// update positions
#ifdef FD_MOVE
	
	// position delta
	vec2 dp = n0.v * timestep;
	
	if (n0.m < 1e3) {
		emitTexel(id[0], vec4(dp, 0.0, 0.0));
	}

	if (should_write_total_speed) {
		emitTexel(nodeLastTexel(), vec4(length(n0.v)));
	}

#endif
	
}

#endif

#ifdef _FRAGMENT_

flat in vec4 vertex_data;
out vec4 frag_data;

void main() {
	frag_data = vertex_data;
}

#endif