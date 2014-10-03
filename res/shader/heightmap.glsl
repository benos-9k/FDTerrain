
#version 330 core

#include "fullscreen.glsl"
#include "nodes.glsl"

#shader fragment

#ifdef _FRAGMENT_

out vec4 frag_color;

vec3 edge_dist(vec2 ep0, vec2 ep1, vec2 p) {
	
	vec2 vx = normalize(ep1 - ep0);
	vec2 vy = cross(vec3(vx, 0.0), vec3(0.0, 0.0, 1.0)).xy;

	// distance along edge from nearest endpoint, negative min inside
	float x0 = abs(dot(p - ep0, vx));
	float x1 = abs(dot(p - ep1, vx));
	float x = mix(-min(x0, x1), min(x0, x1), x0 + x1 > distance(ep0, ep1) + 0.00001); // TODO epsilon?

	// distance parallel to edge
	float y = abs(dot(p - ep0, vy));

	// lerp value
	float z = dot(p - ep0, vx) / distance(ep0, ep1);

	return vec3(x, y, z);

}

void main() {
	
	// current position
	vec2 p = texCoord * 2.0 - 1.0;
	
	float a = 0.0, c = 0.0;

	for (int i = 0; i < num_nodes; i++) {
		Node n0 = nodeGet(i);

		float b = 0.0;

		// 'frequency' multiplier
		float f0 = 9.0 * pow(1.05, n0.d);

		// smoothness exponent
		float s0 = 2.0 - 0.7 * pow(0.95, n0.d);

		// base height from this node
		//float h0 = exp(-pow(f * distance(n0.p, p), s));

		for (int j = 0; j < 4; j++) {
			if (n0.e[j] < 0) continue;
			Node n1 = nodeGet(n0.e[j]);

			float f1 = 9.0 * pow(1.05, n1.d);
			float s1 = 2.0 - 0.7 * pow(0.95, n1.d);

			vec3 ed = edge_dist(n0.p, n1.p, p);

			float x0 = max(ed.x * f0, 0.0);
			float x1 = max(ed.x * f1, 0.0);

			float y0 = ed.y * f0;
			float y1 = ed.y * f1;

			// TODO the heights are coming out wrong somewhere
			//float h0 = pow(0.9, n0.d) * exp(-pow(y0, s0) - pow(x0, s0));
			//float h1 = pow(0.9, n1.d) * exp(-pow(y1, s1) - pow(x1, s1));
			
			//float h = mix(h0, h1, ed.z);

			float l = clamp(ed.z, 0.0, 1.0);
			float h = pow(0.8, mix(n0.d, n1.d, l)) * exp(-pow(mix(y0, y1, l), mix(s0, s1, l)) - pow(mix(x0, x1, l), mix(s0, s1, l)));

			b = max(b, h);
		}

		a = max(a, b);
		//a += b;

		//break;
	}

	frag_color = vec4(vec3(a), 1.0);
}

#endif