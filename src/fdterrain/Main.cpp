
#include <cassert>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <utility>
#include <queue>

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
	float3 p, v, a;
	float m = 1;
	float d = 0;
	unordered_set<Node *> edges;

	inline Node(const float3 &p_) : p(p_) { }

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
		return a + 0.02 * (d + 1.f);
	}

	inline float branch_priority() const {
		return 1.f / max<float>(float(edges.size()) - 2.f + 0.35f * (d + 1.f), 1.f);
	}
	
};

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

// node buffer objects
GLuint nodes0_bo = 0, nodes1_bo = 0;

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

void upload_nodes() {
	if (!nodes0_bo) {
		glGenBuffers(1, &nodes0_bo);
	}
	glBindBuffer(GL_UNIFORM_BUFFER, nodes0_bo);

	vector<GLuint> nodes0_words(nodes.size() * 8);

	for (unsigned i = 0; i < nodes.size(); i++) {
		Node *n = nodes[i];
		reinterpret_cast<float &>(nodes0_words[8 * i + 0]) = n->p.x();
		reinterpret_cast<float &>(nodes0_words[8 * i + 1]) = n->p.y();
		reinterpret_cast<float &>(nodes0_words[8 * i + 2]) = n->v.x();
		reinterpret_cast<float &>(nodes0_words[8 * i + 3]) = n->v.y();
		reinterpret_cast<float &>(nodes0_words[8 * i + 4]) = n->a.x();
		reinterpret_cast<float &>(nodes0_words[8 * i + 5]) = n->a.y();
		reinterpret_cast<float &>(nodes0_words[8 * i + 6]) = n->m;
		reinterpret_cast<float &>(nodes0_words[8 * i + 7]) = n->d;
	}

	glBufferData(GL_UNIFORM_BUFFER, nodes0_words.size() * sizeof(GLuint), &nodes0_words[0], GL_DYNAMIC_DRAW);

}

void download_nodes() {

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
	nodes.push_back(n0);
	nodes.push_back(n1);
	make_edge(n0, n1);
}

void step() {
	if (ek_avg < 0.5) {
		// one generation finished, prepare next one
		
		uniform_int_distribution<unsigned> bd(0, 1), cd(0, 100);

		// make 'old' active nodes heavier, and remove the heaviest
		vector<Node *> old_active_nodes = move(active_nodes);
		for (Node *n : old_active_nodes) {
			n->m *= 1.2;
			if (n->m < 300) {
				active_nodes.push_back(n);
			}
		}

		// subdivide some edges (depth is averaged)
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
			uniform_real_distribution<float> lerp_dist(0, 1);
			Node *n2 = new Node(float3::lerp(n0->p, n1->p, lerp_dist(ran0)));
			n2->d = 0.5f * (n0->d + n1->d);
			if (cd(ran0) > 85) {
				n2->d *= 0.75;
			}
			split_edge(n0, n2, n1);
			nodes.push_back(n2);
			active_nodes.push_back(n2);
		}

		// make some branches (depth + 1)
		priority_queue<node_branch_ptr> branch_q;
		for (Node *n : nodes) {
			branch_q.push(n);
		}
		for (unsigned i = 0; i < nodes.size() / 6 + 1; i++) {
			Node *n0 = branch_q.top().get();
			branch_q.pop();
			if (bd(ran0)) continue;
			uniform_real_distribution<float> pd(-0.1, 0.1);
			Node *n2 = new Node(n0->p + float3(pd(ran0), pd(ran0), 0));
			n2->d = n0->d + 1.f;
			make_edge(n0, n2);
			nodes.push_back(n2);
			active_nodes.push_back(n2);
		}

	}

	float ek = 0;

	// calculate forces, accelerations, velocities, ek
	for (Node *n0 : active_nodes) {
		// acting force
		float3 f;

		float3 fff = float3(-n0->p.x(), -n0->p.y(), 0.f);
		fff *= 0.65;
		fff = fff * fff * fff;
		f += float3(fff.x() * 200, fff.y() * 200, 0) * nodes.size();

		// charge repulsion from every node
		for (Node *n1 : nodes) {
			if (n1 == n0) continue;
			// direction is away from other node
			float3 v = n0->p - n1->p;
			float d2 = float3::dot(v, v);
			float k = min((1 / d2) * 70.f, 100000.f); // charge constant
			float3 fc = v.unit() * k;
			if (fc.isnan()) {
				uniform_real_distribution<float> fd(-1, 1);
				f += float3(fd(ran0), fd(ran0), 0);
			} else {
				f += fc;
			}
		}
		
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
		ek += 0.5f * n0->m * pow(n0->v.mag(), 2.f);
	}

	ek_avg = ek / active_nodes.size();

	// update position
	for (Node *n0 : active_nodes) {
		n0->p += n0->v * 0.0001; // timestep
	}


}

void display(const size2i &sz) {

	static const int hmap_size = 512;

	static const int max_nodes = 2048;

	static auto prog_hmap_spec = shader_program_spec().source("heightmap.glsl").define("MAX_NODES", max_nodes);

	assert(nodes.size() <= max_nodes);

	GLuint prog_hmap = win->shaderManager()->program(prog_hmap_spec);

	upload_nodes();

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

	glBindBufferBase(GL_UNIFORM_BUFFER, 0, nodes0_bo);
	glUniformBlockBinding(prog_hmap, glGetUniformBlockIndex(prog_hmap, "NodesBlock"), 0);

	glUniform1i(glGetUniformLocation(prog_hmap, "num_nodes"), nodes.size());

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

	while (!win->shouldClose()) {
		glfwPollEvents();

		auto frame_start_time = chrono::steady_clock::now();
		while (chrono::steady_clock::now() - frame_start_time < chrono::milliseconds(15)) {
			step();
		}

		display(win->size());

		ostringstream title;
		title << "Force Directed Terrain [" << nodes.size() << " nodes]";
		win->title(title.str());

		glFinish();
		win->swapBuffers();
	}

	delete win;
	glfwTerminate();

}
