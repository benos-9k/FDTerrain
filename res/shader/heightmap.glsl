
#version 330 core

#include "fullscreen.glsl"

#shader fragment

#ifndef MAX_NODES
#define MAX_NODES 1024
#endif

struct Node {
	// this should be 2x vec4 in std140
	// TODO edges
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

		x *= 12.0 + 3.0 * n.d;
		
		float b = (0.9 * pow(0.6, n.d) + 0.1) * exp(-pow(abs(1.0 * x), 2.0 - 0.9 * pow(0.7, n.d)));

		a += b;
	}

	frag_color = vec4(vec3(a), 1.0);
}

#endif