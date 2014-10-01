
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

		x *= 12.0 * pow(1.3, n.d);
		
		float b = pow(0.5, n.d) * exp(-pow(abs(x), 2.0 - 0.8 * pow(0.7, n.d)));

		a += b;
	}

	frag_color = vec4(vec3(a), 1.0);
}

#endif