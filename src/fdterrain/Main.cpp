
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <utility>
#include <queue>
#include <algorithm>
#include <stdexcept>

#include <gecom/Window.hpp>
#include <gecom/Chrono.hpp>

#include "Float3.hpp"


using namespace std;
using namespace gecom;
using namespace i3d;

Window *win = nullptr;

double rotx = math::pi() / 6;
double roty = math::pi() / 4;
double dolly = -2.5;


struct Node {
	static unsigned next_id;

	float3 p, v, a;
	float m = 1;
	float d = 0;
	unordered_set<Node *> edges;

	// this is used as an index into the nodes vector
	// kinda convoluted, but screw it
	unsigned id;

	inline Node(const float3 &p_) : p(p_), id(next_id++) { }

	inline float charge() {
		// TODO tweak
		return max<float>(8.f + 0.0f * d, 2.f);
	}

	inline Node * edge(unsigned i) const {
		for (auto it = edges.begin(); it != edges.end(); it++) {
			if (i == 0) return *it;
			i--;
		}
		return nullptr;
	}

	inline float split_priority() const {
		float a = 0.f;
		for (Node *n : edges) {
			a += (n->p - p).mag();
		}
		// dont divide by num edges, works better
		// return a + 0.01 * d;
		return a + 0.02 * (d + 1.f);
	}

	inline float branch_priority() const {
		// TODO how does this work?
		// TODO leaves should not branch?
		//if (edges.size() <= 1) return 0.f;
		//return 1.f / float(edges.size()) - 0.07 * d;
		return 1.f / max<float>(float(edges.size()) - 2.f + 0.35f * (d + 1.f), 1.f);
	}
	
};

unsigned Node::next_id = 0;

// Barnes-Hut quadtree for charge repulsion
// http://www.cs.princeton.edu/courses/archive/fall03/cs126/assignments/barnes-hut.html
class bh_tree {
private:
	using set_t = unordered_set<Node *>;

	class bh_node {
	private:
		aabb m_bound;
		float3 m_coc;
		bh_node *m_children[4];
		set_t m_values;
		size_t m_count = 0;
		float m_charge = 0;
		bool m_isleaf = true;

		inline unsigned childID(const float3 &p) const {
			return 0x3 & _mm_movemask_ps((p >= m_bound.center()).data());
		}

		// mask is true where cid bit is _not_ set
		inline __m128 childInvMask(unsigned cid) const {
			__m128i m = _mm_set1_epi32(cid);
			m = _mm_and_si128(m, _mm_set_epi32(0, 4, 2, 1));
			m = _mm_cmpeq_epi32(_mm_setzero_si128(), m);
			return _mm_castsi128_ps(m);
		}

		inline aabb childBound(unsigned cid) const {
			// positive / negative halfsizes
			__m128 h = m_bound.halfsize().data();
			__m128 g = _mm_sub_ps(_mm_setzero_ps(), h);

			// convert int bitmask to (opposite) sse mask
			__m128 n = childInvMask(cid);

			// vector to a corner of the current node's aabb
			float3 vr(_mm_or_ps(_mm_and_ps(n, g), _mm_andnot_ps(n, h)));
			const float3 c = m_bound.center();

			return aabb::fromPoints(c, c + vr);
		}

		inline void unleafify();

	public:
		inline bh_node(const aabb &a_) : m_bound(a_), m_coc(a_.center()) {
			// clear child pointers
			std::memset(m_children, 0, 4 * sizeof(bh_node *));
		}

		bh_node(const bh_node &) = delete;
		bh_node & operator=(const bh_node &) = delete;

		inline aabb bound() {
			return m_bound;
		}

		inline size_t count() {
			return m_count;
		}

		inline bool insert(Node *n, bool reinsert = false) {
			if (m_isleaf && m_count < 8) {
				if (!m_values.insert(n).second) return false;
			} else {
				// not a leaf or should not be
				unleafify();
				unsigned cid = childID(n->p);
				// element contained in one child node (its a point) - create if necessary then insert
				bh_node *child = m_children[cid];
				if (!child) {
					child = new bh_node(childBound(cid));
					m_children[cid] = child;
				}
				if (!child->insert(n)) return false;
			}
			// allow re-inserting internally to skip accumulation
			if (reinsert) return true;
			m_count++;
			// update charge and centre-of-charge
			m_coc = (m_coc * m_charge + n->p * n->charge()) / (m_charge + n->charge());
			m_charge += n->charge();
			return true;
		}

		inline float3 force(Node *n0) {

			// can we treat this node as one charge?
			// compare bound width to distance from node to centre-of-charge
			{
				// direction is away from coc
				float3 v = n0->p - m_coc;
				float id2 = 1.f / float3::dot(v, v);
				float s = m_bound.halfsize().x() + m_bound.halfsize().y();
				float q2 = s * s * id2;
				// note that this is the square of the ratio of interest
				// too much higher and it doesnt converge very well
				if (q2 < 0.5) {
					float k = min(id2 * n0->charge() * m_charge, 100000.f);
					// shouldnt need to nan check
					return v.unit() * k;
				}
			}

			float3 f(0);
			
			// force from nodes in this node
			for (Node *n1 : m_values) {
				if (n1 == n0) continue;
				// direction is away from other node
				float3 v = n0->p - n1->p;
				float id2 = 1.f / float3::dot(v, v);
				float k = min(id2 * n0->charge() * n1->charge(), 100000.f);
				float3 fc = v.unit() * k;
				if (fc.isnan()) {
					f += float3(0, 0.1, 0);
				} else {
					f += fc;
				}
			}

			// recurse
			for (bh_node **pn = m_children + 4; pn --> m_children; ) {
				if (*pn) {
					f += (*pn)->force(n0);
				}
			}
			
			return f;
		}

		inline ~bh_node() {
			for (bh_node **pn = m_children + 4; pn --> m_children; ) {
				if (*pn) delete *pn;
			}
		}

	};

	bh_node *m_root = nullptr;

	// kill the z dimension of an aabb so this actually functions as a quadtree
	static inline aabb sanitize(const aabb &a) {
		float3 c = a.center();
		float3 h = a.halfsize();
		return aabb(float3(c.x(), c.y(), 0), float3(h.x(), h.y(), 0));
	}

	inline void destroy() {
		if (m_root) delete m_root;
		m_root = nullptr;
	}

public:
	inline bh_tree() { }

	inline bh_tree(const aabb &rootbb) {
		m_root = new bh_node(sanitize(rootbb));
	}

	inline bh_tree(const bh_tree &other) {
		assert(false && "not implemented yet");
	}

	inline bh_tree(bh_tree &&other) {
		m_root = other.m_root;
		other.m_root = nullptr;
	}

	inline bh_tree & operator=(const bh_tree &other) {
		assert(false && "not implemented yet");
		return *this;
	}

	inline bh_tree & operator=(bh_tree &&other) {
		destroy();
		m_root = other.m_root;
		other.m_root = nullptr;
		return *this;
	}

	inline bool insert(Node *n) {
		if (!m_root) m_root = new bh_node(aabb(float3(0), float3(1, 1, 0)));
		if (m_root->bound().contains(n->p)) {
			return m_root->insert(n);
		} else {
			assert(false && "not implemented yet");
			return false;
		}
	}

	inline float3 force(Node *n0) {
		if (!m_root) return float3(0);
		return m_root->force(n0);
	}

	inline ~bh_tree() {
		destroy();
	}

};

inline void bh_tree::bh_node::unleafify() {
	if (m_isleaf) {
		m_isleaf = false;
		set_t temp = move(m_values);
		for (Node *n : temp) {
			insert(n, true);
		}
	}
}

class node_ptr {
private:
	Node *m_ptr;

public:
	node_ptr(Node *ptr) : m_ptr(ptr) { }

	Node * get() const {
		return m_ptr;
	}

	Node & operator*() {
		return *m_ptr;
	}

	const Node & operator*() const {
		return *m_ptr;
	}

	Node * operator->() {
		return m_ptr;
	}

	const Node * operator->() const {
		return m_ptr;
	}
};

// TODO this shit is weird, stop being silly
// or is it? need to cache priorities before queueing

class node_split_ptr : public node_ptr {
private:
	float m_x;

public:
	node_split_ptr(Node *n) : node_ptr(n), m_x(n->split_priority()) { }
	
	bool operator<(const node_split_ptr &n) const {
		return m_x < n.m_x;
	}
};

class node_branch_ptr : public node_ptr {
private:
	float m_x;

public:
	node_branch_ptr(Node *n) : node_ptr(n), m_x(n->branch_priority()) { }

	bool operator<(const node_branch_ptr &n) const {
		return m_x < n.m_x;
	}
};

vector<Node *> nodes;
vector<Node *> active_nodes;
float ek_avg = -1;

// change to avoid name conflict - there is a function random() in gcc stdlib.h
std::default_random_engine ran0;

// node textures
GLuint tex_nodes_p = 0;
GLuint tex_nodes_vmd = 0;
GLuint tex_nodes_e = 0;

// fbo + texture for heightmap
GLuint fbo_hmap = 0;
GLuint tex_hmap = 0;

mat4d perspectiveFOV(double fov_y, double aspect, double znear, double zfar) {
	mat4d m(0);
	double f = math::cot(fov_y / 2.0);
	m(0, 0) = f / aspect;
	m(1, 1) = f;
	m(2, 2) = (zfar + znear) / (znear - zfar);
	m(2, 3) = (2 * zfar * znear) / (znear - zfar);
	m(3, 2) = -1;
	return m;
}

void draw_fullscreen(unsigned instances = 1) {
	static GLuint vao = 0;
	if (vao == 0) {
		glGenVertexArrays(1, &vao);
	}
	glBindVertexArray(vao);
	glDrawArraysInstanced(GL_POINTS, 0, 1, instances);
	glBindVertexArray(0);
}

void set_node_uniforms(GLuint prog) {
	// bind textures
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_nodes_p);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, tex_nodes_vmd);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, tex_nodes_e);
	// now set uniforms
	glUniform1i(glGetUniformLocation(prog, "num_nodes"), nodes.size());
	glUniform1i(glGetUniformLocation(prog, "sampler_p"), 0);
	glUniform1i(glGetUniformLocation(prog, "sampler_vmd"), 1);
	glUniform1i(glGetUniformLocation(prog, "sampler_e"), 2);

}

void upload_nodes() {
	
	// 128 => 16k max nodes
	static const unsigned tex_size = 128;

	if (nodes.size() > tex_size * tex_size - 1) {
		// last texel is reserved
		throw runtime_error("too many nodes you twit");
	}

	glActiveTexture(GL_TEXTURE0);

	if (!tex_nodes_p) {
		glGenTextures(1, &tex_nodes_p);
		glBindTexture(GL_TEXTURE_2D, tex_nodes_p);
		// nvidia uses this as mipmap allocation hint; not doing it causes warning spam
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	if (!tex_nodes_vmd) {
		glGenTextures(1, &tex_nodes_vmd);
		glBindTexture(GL_TEXTURE_2D, tex_nodes_vmd);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	if (!tex_nodes_e) {
		glGenTextures(1, &tex_nodes_e);
		glBindTexture(GL_TEXTURE_2D, tex_nodes_e);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	vector<float> data_p(2 * tex_size * tex_size);
	vector<float> data_vmd(4 * tex_size * tex_size);
	vector<GLshort> data_e(4 * tex_size * tex_size);

	for (unsigned i = 0; i < nodes.size(); i++) {
		Node *n = nodes[i];
		data_p[2 * i + 0] = n->p.x();
		data_p[2 * i + 1] = n->p.y();
		data_vmd[4 * i + 0] = n->v.x();
		data_vmd[4 * i + 1] = n->v.y();
		data_vmd[4 * i + 2] = n->m;
		data_vmd[4 * i + 3] = n->d;
		// set the edges that exist and mark those that don't
		for (unsigned j = 0; j < 4; j++) {
			Node *e = n->edge(j);
			if (e) {
				data_e[4 * i + j] = e->id;
			} else {
				data_e[4 * i + j] = -1;
			}
		}
	}

	// 0 the last texel in VMD so it accumulates speeds correctly
	fill(data_vmd.end() - 4, data_vmd.end(), 0);

	glBindTexture(GL_TEXTURE_2D, tex_nodes_p);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, tex_size, tex_size, 0, GL_RG, GL_FLOAT, &data_p[0]);

	glBindTexture(GL_TEXTURE_2D, tex_nodes_vmd);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tex_size, tex_size, 0, GL_RGBA, GL_FLOAT, &data_vmd[0]);

	glBindTexture(GL_TEXTURE_2D, tex_nodes_e);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16I, tex_size, tex_size, 0, GL_RGBA_INTEGER, GL_SHORT, &data_e[0]);

	glBindTexture(GL_TEXTURE_2D, 0);

	
}

void download_nodes() {

	static const unsigned tex_size = 128;

	vector<float> data_p(2 * tex_size * tex_size);
	vector<float> data_vmd(4 * tex_size * tex_size);
	vector<GLshort> data_e(4 * tex_size * tex_size);

	glActiveTexture(GL_TEXTURE0);

	glBindTexture(GL_TEXTURE_2D, tex_nodes_p);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, &data_p[0]);

	glBindTexture(GL_TEXTURE_2D, tex_nodes_vmd);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &data_vmd[0]);

	// edges arent modified by gpu so no need to download

	for (unsigned i = 0; i < nodes.size(); i++) {
		Node *n = nodes[i];
		float3 p(data_p[2 * i + 0], data_p[2 * i + 1], 0);
		n->p = p;
		float3 v(data_vmd[4 * i + 0], data_vmd[4 * i + 1], 0);
		n->v = v;
		// mass and depth shouldnt be modified
		//n->m = data_vmd[4 * i + 2];
		//n->d = data_vmd[4 * i + 3];
	}

}

void make_edge(Node *a, Node *b) {
	a->edges.insert(b);
	b->edges.insert(a);
}

void split_edge(Node *a, Node *b, Node *c) {
	a->edges.erase(c);
	c->edges.erase(a);
	make_edge(a, b);
	make_edge(b, c);
}

void init() {
	Node *n0, *n1;
	n0 = new Node(float3(-1, 0, 0));
	n1 = new Node(float3(1, 0, 0));
	n0->m = 1e3;
	n1->m = 1e3;
	nodes.push_back(n0);
	nodes.push_back(n1);
	make_edge(n0, n1);
	upload_nodes();
}

void subdivide_and_branch() {
	// one generation finished, prepare next one

	if (nodes.size() > 32) return;
	
	uniform_int_distribution<unsigned> bd(0, 1), cd(0, 100);

	// make 'old' active nodes heavier, and remove the heaviest
	vector<Node *> old_active_nodes = move(active_nodes);
	for (Node *n : old_active_nodes) {
		n->m *= 1.2;
		if (n->m < 300) {
			active_nodes.push_back(n);
		}
	}

	// subdivide some edges (depth is averaged then randomly modified)
	priority_queue<node_split_ptr> split_q;
	for (Node *n : nodes) {
		split_q.push(n);
	}
	for (unsigned i = 0; i < nodes.size() / 15 + 1; i++) {
		Node *n0 = split_q.top().get();
		split_q.pop();
		if (bd(ran0)) continue;
		priority_queue<node_split_ptr> q2;
		for (Node *n : n0->edges) {
			q2.push(n);
		}
		Node *n1 = q2.top().get();
		assert(n1);
		uniform_real_distribution<float> fd(0, 1);
		Node *n2 = new Node(float3::lerp(n0->p, n1->p, fd(ran0)));
		n2->d = 0.5f * (n0->d + n1->d);
		n2->d += 0.5 * fd(ran0) - 0.3;
		split_edge(n0, n2, n1);
		nodes.push_back(n2);
		active_nodes.push_back(n2);
	}

	// make some branches (depth + 1)
	priority_queue<node_branch_ptr> branch_q;
	for (Node *n : nodes) {
		branch_q.push(n);
	}
	for (unsigned i = 0; i < nodes.size() / 60000 + 0; i++) {
		Node *n0 = branch_q.top().get();
		branch_q.pop();
		// max allowed edges is 4
		if (n0->edges.size() >= 4) continue;
		if (bd(ran0)) continue;
		uniform_real_distribution<float> pd(-0.1, 0.1);
		Node *n2 = new Node(n0->p + float3(pd(ran0), pd(ran0), 0));
		n2->d = n0->d + 1;
		make_edge(n0, n2);
		nodes.push_back(n2);
		active_nodes.push_back(n2);
	}

}

unsigned step_gpu() {

	throw logic_error("gpu implementation is deprecated");

	if (ek_avg < 1.5) {
		download_nodes();
		subdivide_and_branch();
		upload_nodes();
	}

	static const unsigned tex_size = 128;

	static auto prog_update_spec = shader_program_spec().source("forcedirection.glsl").define("FD_UPDATE");
	static auto prog_move_spec = shader_program_spec().source("forcedirection.glsl").define("FD_MOVE");

	GLuint prog_update = win->shaderManager()->program(prog_update_spec);
	GLuint prog_move = win->shaderManager()->program(prog_move_spec);

	static GLuint fbo = 0;
	if (!fbo) {
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_nodes_p, 0);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, tex_nodes_vmd, 0);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glViewport(0, 0, tex_size, tex_size);

	// add things
	glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	// damping constant
	glBlendColor(0.995, 0.995, 0.995, 1.0);

	// load/store velocity accumulator texel
	float ekxx[] { 0, 0, 0, 0 };

	// zero velocity accumulator
	glBindTexture(GL_TEXTURE_2D, tex_nodes_p);
	glTexSubImage2D(GL_TEXTURE_2D, 0, tex_size - 1, tex_size - 1, 1, 1, GL_RG, GL_FLOAT, ekxx);

	
	static const unsigned iterations = 10;
	for (unsigned i = 0; i < iterations; i++) {

		// write total speed on last iteration
		bool should_write_total_speed = (i + 1) == iterations;

		// update velocity
		glDrawBuffer(GL_COLOR_ATTACHMENT1);
		// prevent writing to mass/depth
		glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_FALSE);
		glUseProgram(prog_update);
		// use damping blend
		glBlendFuncSeparate(GL_ONE, GL_CONSTANT_COLOR, GL_ONE, GL_ONE);
		set_node_uniforms(prog_update);
		draw_fullscreen(nodes.size());

		// update position
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glUseProgram(prog_move);
		// use hard blend
		glBlendFunc(GL_ONE, GL_ONE);
		glUniform1i(glGetUniformLocation(prog_move, "should_write_total_speed"), should_write_total_speed);
		set_node_uniforms(prog_move);
		draw_fullscreen(nodes.size());

	}

	glReadPixels(tex_size - 1, tex_size - 1, 1, 1, GL_RG, GL_FLOAT, ekxx);
	//log() << ekxx[0];
	ek_avg = ekxx[0] / nodes.size();

	glDisable(GL_BLEND);

	glFinish();

	return iterations;
}

unsigned step() {
	
	if (ek_avg < 2.0) {
		subdivide_and_branch();
	}

	float ek = 0;

	// assemble barnes-hut tree for charge repulsion forces
	// NOTE this is very slow with VS debugger attached, even a release-mode build
	bh_tree bht(aabb(float3(0), float3(3, 3, 0)));
	for (Node *n : nodes) {
		bht.insert(n);
	}

	// calculate forces, accelerations, velocities, ek
#pragma omp parallel for
	for (int i = 0; i < active_nodes.size(); i++) {
		Node *n0 = active_nodes[i];

		// acting force
		float3 f;

		float3 fff = float3(-n0->p.x(), -n0->p.y(), 0.f);
		fff *= 0.65;
		fff = fff * fff * fff;
		f += float3(fff.x() * 200, fff.y() * 200, 0) * nodes.size();

		// charge repulsion from every node
		f += bht.force(n0);

		// spring contraction from connected nodes
		for (Node *n1 : n0->edges) {
			// direction is towards other node
			float3 v = n1->p - n0->p;
			f += 1000000.f * v; // spring constant
		}

		// acceleration
		n0->a = f / n0->m;

		// velocity
		n0->v += n0->a * 0.0001; // timestep

		// damping
		n0->v *= 0.995;

		// ek
		// TODO this is a race condition
		ek += n0->v.mag();
	}

	ek_avg = ek / active_nodes.size();

	// update position
#pragma omp parallel for
	for (int i = 0; i < active_nodes.size(); i++) {
		Node *n0 = active_nodes[i];
		n0->p += n0->v * 0.0001; // timestep
	}

	return 1;
}

void display(const size2i &sz) {

	static const int hmap_size = 512;

	static auto prog_hmap_spec = shader_program_spec().source("heightmap.glsl");

	GLuint prog_hmap = win->shaderManager()->program(prog_hmap_spec);

	if (!tex_hmap) {
		glGenTextures(1, &tex_hmap);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex_hmap);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, hmap_size, hmap_size, 0, GL_RED, GL_FLOAT, nullptr);
	}

	if (!fbo_hmap) {
		glGenFramebuffers(1, &fbo_hmap);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_hmap);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_hmap, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
	}

	// first draw the heightmap

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_hmap);
	glViewport(0, 0, hmap_size, hmap_size);
	
	glUseProgram(prog_hmap);

	set_node_uniforms(prog_hmap);

	draw_fullscreen();

	// now draw terrain

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glClearColor(1, 1, 1, 1);
	glViewport(0, 0, sz.w, sz.h);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_hmap);
	glGenerateMipmap(GL_TEXTURE_2D);

	static auto prog_terr_spec = shader_program_spec().source("terrain.glsl");

	GLuint prog_terr = win->shaderManager()->program(prog_terr_spec);

	glUseProgram(prog_terr);

	glUniform1i(glGetUniformLocation(prog_terr, "sampler_hmap"), 0);

	mat4d mv = mat4d::translate(0, 0, dolly) * mat4d::rotateX(rotx) * mat4d::rotateY(roty);
	mat4d proj = perspectiveFOV(math::pi() / 3, sz.ratio(), 0.1, 100);

	glUniformMatrix4fv(glGetUniformLocation(prog_terr, "modelview_matrix"), 1, true, mat4f(mv));
	glUniformMatrix4fv(glGetUniformLocation(prog_terr, "projection_matrix"), 1, true, mat4f(proj));

	glEnable(GL_DEPTH_TEST);

	draw_fullscreen((hmap_size - 1) * (hmap_size - 1));

	glDisable(GL_DEPTH_TEST);

	glUseProgram(0);

}

void display_old(const size2i &sz) {
	glClearColor(1, 1, 1, 1);

	glViewport(0, 0, sz.w, sz.h);
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glColor3f(1, 0, 0);
	for (Node *n0 : nodes) {
		for (Node *n1 : n0->edges) {
			glLineWidth(15.f / (0.5f * (n0->d + n1->d) + 2.5f));
			glBegin(GL_LINES);
			glVertex3f(n0->p.x(), n0->p.y(), 0);
			glVertex3f(n1->p.x(), n1->p.y(), 0);
			glEnd();
		}
	}

	glColor3f(0, 0, 1);
	glPointSize(5);
	glBegin(GL_POINTS);
	for (Node *n : nodes) {
		glVertex3f(n->p.x(), n->p.y(), 0);
	}
	glEnd();
}

int main() {

	win = createWindow().visible(true).size(768, 768).title("Force Directed Terrain");
	win->makeContextCurrent();

	GLint max_uniform_block_size;
	glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &max_uniform_block_size);
	log() << "max uniform block size: " << max_uniform_block_size;

	win->shaderManager()->addSourceDirectory("./res/shader");

	win->onKeyPress.subscribe([](const key_event &e) {
		if (e.key == GLFW_KEY_RIGHT) {
			roty -= 0.02;
		}
		if (e.key == GLFW_KEY_LEFT) {
			roty += 0.02;
		}
		if (e.key == GLFW_KEY_UP) {
			rotx += 0.02;
		}
		if (e.key == GLFW_KEY_DOWN) {
			rotx -= 0.02;
		}
		if (e.key == GLFW_KEY_EQUAL) {
			dolly += 0.02;
		}
		if (e.key == GLFW_KEY_MINUS) {
			dolly -= 0.02;
		}
		return false;
	}).forever();

	init();

	auto time_last_fps = chrono::steady_clock::now();
	unsigned sps = 0;

	while (!win->shouldClose()) {
		glfwPollEvents();


		auto frame_start_time = chrono::steady_clock::now();
		while (chrono::steady_clock::now() - frame_start_time < chrono::milliseconds(30)) {
			sps += step();
			// sps += step_gpu();
		}

		// if using step() need to upload before draw
		upload_nodes();
		display(win->size());

		if (chrono::steady_clock::now() - time_last_fps > chrono::seconds(1)) {
			time_last_fps = chrono::steady_clock::now();

			ostringstream title;
			title << "Force Directed Terrain [" << nodes.size() << " nodes, " << sps << " SPS, " << ek_avg << " EKavg]";
			win->title(title.str());

			sps = 0;
		}

		glFinish();
		win->swapBuffers();
	}

	delete win;
	glfwTerminate();

}
