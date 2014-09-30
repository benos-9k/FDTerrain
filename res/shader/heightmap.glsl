
#version 330 core

#include "fullscreen.glsl"
#include "nodes.glsl"

#shader fragment

#ifdef _FRAGMENT_

out vec4 frag_color;

void main() {
	
	float a = 0.0;

	for (int i = 0; i < num_nodes; i++) {
		Node n = nodeGet(i);

		float x = distance(n.p, texCoord * 2.0 - 1.0);

		x *= 12.0 + 3.0 * n.d;
		
		float b = (0.9 * pow(0.6, n.d) + 0.1) * exp(-pow(abs(1.0 * x), 2.0 - 0.8 * pow(0.7, n.d)));

		a += b;
	}

	frag_color = vec4(vec3(a), 1.0);
}

#endif