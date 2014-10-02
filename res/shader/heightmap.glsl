
#version 330 core

#include "fullscreen.glsl"
#include "nodes.glsl"

#shader fragment

#ifdef _FRAGMENT_

out vec4 frag_color;

vec2 edge_dist(vec2 ep0, vec2 ep1, vec2 p) {
	
	vec2 vx = normalize(ep1 - ep0);
	vec2 vy = cross(vec3(vx, 0.0), vec3(0.0, 0.0, 1.0)).xy;

	// distance along edge from nearest endpoint, negative min inside
	float x0 = abs(dot(p - ep0, vx));
	float x1 = abs(dot(p - ep1, vx));
	float x = mix(-min(x0, x1), min(x0, x1), x0 + x1 > distance(ep0, ep1) + 0.00001); // TODO epsilon?

	// distance parallel to edge
	float y = abs(dot(p - ep0, vy));

	return vec2(x, y);

}

void main() {
	
	float a = 0.0;

	for (int i = 0; i < num_nodes; i++) {
		Node n0 = nodeGet(i);

		float b = 0.0, c = 0.0;

		for (int j = 0; j < 4; j++) {
			if (n0.e[j] < 0) continue;
			Node n1 = nodeGet(n0.e[j]);

			vec2 ed = edge_dist(n0.p, n1.p, texCoord * 2.0 - 1.0);

			float x = max(ed.x * 36.0 + 0.89, 0.0);

			float y = ed.y * 12.0;

			float h = pow(0.5, n0.d) * exp(-pow(x, 3.0)) * exp(-pow(abs(y), 2.0 - 0.8 * pow(0.7, n0.d)));
			
			b += h;
			c += 1.0;

		}

		a += b;

		//float x = distance(n.p, texCoord * 2.0 - 1.0);
		//x *= 12.0 * pow(1.3, n.d);
		//float b = pow(0.5, n.d) * exp(-pow(abs(x), 2.0 - 0.8 * pow(0.7, n.d)));
		//a += b;
	}

	frag_color = vec4(vec3(a), 1.0);
}

#endif