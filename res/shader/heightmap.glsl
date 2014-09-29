
#version 330 core

#include "fullscreen.glsl"

#shader fragment

#ifndef MAX_NODES
#define MAX_NODES 1024
#endif

struct Node {
	// this should be 2x vec4 in std140
	vec2 p, v, a;
	float m, d;
};

layout(std140) uniform NodesBlock {
	Node nodes[MAX_NODES];
};

uniform int num_nodes;

#ifdef _FRAGMENT_

out vec4 frag_color;

void main() {
	
	float a = 0.0;

	for (int i = 0; i < num_nodes; i++) {
		Node n = nodes[i];

		float x = distance(n.p, texCoord * 2.0 - 1.0);

		x *= 8.0;
		
		a += exp(-n.d) * exp(-pow(abs(1.0 * x), 2.0)) * (cos(1.0 * x) + 0.8) / 1.8;

	}

	//a /= 2.0;

	frag_color = vec4(vec3(a), 1.0);
}

#endif