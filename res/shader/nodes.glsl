
// texture-based node representation stuff
#version 330 core

// position
// write position changes using additive blending
uniform sampler2D sampler_p;

// velocity, mass, depth
// write velocity changes using additive blending
// also write all speeds to one texel to calc total speed (when requested!)
uniform sampler2D sampler_vmd;

// edges
// this isnt written to by shaders
uniform isampler2D sampler_e;

// total number of nodes
uniform int num_nodes;

struct Node {
	vec2 p, v;
	float m, d;
	ivec4 e;
};

ivec2 nodeLastTexel() {
	return textureSize(sampler_p, 0) - 1;
}

ivec2 nodeTexelFromID(int id) {
	ivec2 ts = textureSize(sampler_p, 0);
	return ivec2(id % ts.x, id / ts.x);
}

vec2 nodeTexCoord(ivec2 tx) {
	vec2 ts = vec2(textureSize(sampler_p, 0));
	return (vec2(tx) + 0.5) / ts;
}

vec2 nodeTexCoord(int id) {
	return nodeTexCoord(nodeTexelFromID(id));
}

Node nodeGet(ivec2 tx) {
	Node n;
	n.p = texelFetch(sampler_p, tx, 0).xy;
	vec4 temp = texelFetch(sampler_vmd, tx, 0);
	n.v = temp.xy;
	n.m = temp.z;
	n.d = temp.w;
	n.e = texelFetch(sampler_e, tx, 0);
	return n;
}

Node nodeGet(int id) {
	return nodeGet(nodeTexelFromID(id));
}
