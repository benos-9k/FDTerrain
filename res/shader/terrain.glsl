
#version 330 core

#shader vertex
#shader geometry
#shader fragment

uniform sampler2D sampler_hmap;
uniform mat4 modelview_matrix;
uniform mat4 projection_matrix;

#ifdef _VERTEX_

flat out int id;

void main() {
	id = gl_InstanceID;
}

#endif

#ifdef _GEOMETRY_

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

flat in int id[];

out VertexData {
	vec3 pos_v;
	vec3 norm_v;
	vec3 norm_w;
} vertex_out;

// mean height
float hh;

vec3 positionFromTexel(ivec2 tx) {
	ivec2 ts = textureSize(sampler_hmap, 0);
	// texelFetch() is undefined when out of bounds - clamp to edges
	tx = clamp(tx, ivec2(0), ts - 1);
	float h = texelFetch(sampler_hmap, tx, 0).r * 0.07 / hh;
	return vec3(vec2(tx) / (vec2(ts) - 1.0) * 2.0 - 1.0, h).xzy;
}

vec3 normalFromTexel(ivec2 tx) {
	vec3 p0 = positionFromTexel(tx);
	vec3 n = vec3(0.0);
	const ivec2[] dp = ivec2[](ivec2(1, 0), ivec2(0, -1), ivec2(-1, 0), ivec2(0, 1));
	for (int i = 0; i < 4; i++) {
		n += normalize(cross(normalize(positionFromTexel(tx + dp[i]) - p0), normalize(positionFromTexel(tx + dp[(i + 1) % 4]) - p0)));
	}
	return normalize(n);
}

void main() {
	
	// get mean height from mipmap to scale real heights
	// the mean height to start with is very low because low node count,
	// which is why this intially over-compensates but slowly settles down
	hh = textureLod(sampler_hmap, vec2(0.5), 9001.0).r;

	ivec2 ts = textureSize(sampler_hmap, 0);
	ivec2 tx0 = ivec2(id[0] % (ts.x - 1), id[0] / (ts.x - 1));

	const ivec2[] dp = ivec2[](ivec2(0, 0), ivec2(1, 0), ivec2(0, 1), ivec2(1, 1));

	for (int i = 0; i < 4; i++) {
		
		vec3 p = positionFromTexel(tx0 + dp[i]);
		vec3 n = normalFromTexel(tx0 + dp[i]);

		vertex_out.pos_v = (modelview_matrix * vec4(p, 1.0)).xyz;
		vertex_out.norm_v = (modelview_matrix * vec4(n, 0.0)).xyz;
		vertex_out.norm_w = n;

		gl_Position = projection_matrix * modelview_matrix * vec4(p, 1.0);
		EmitVertex();

	}

	EndPrimitive();

}

#endif

#ifdef _FRAGMENT_

in VertexData {
	vec3 pos_v;
	vec3 norm_v;
	vec3 norm_w;
} vertex_in;

out vec4 frag_color;

void main() {
	vec3 n = normalize(vertex_in.norm_w);

	frag_color = vec4(0.0, pow(n.y * 0.8, 1.5), 0.0, 1.0);
}

#endif
