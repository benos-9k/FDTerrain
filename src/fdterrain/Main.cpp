
#include <cassert>
#include <iostream>
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

	Node * get() {
		return m_ptr;
	}

	const Node * get() const {
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
class node_split_ptr : public node_ptr {
public:
	node_split_ptr(Node *n) : node_ptr(n) { }
	
	bool operator<(const node_split_ptr &n) const {
		return get()->split_priority() < n->split_priority();
	}
};

class node_branch_ptr : public node_ptr {
public:
	node_branch_ptr(Node *n) : node_ptr(n) { }

	bool operator<(const node_branch_ptr &n) const {
		return get()->branch_priority() < n->branch_priority();
	}
};

vector<Node *> nodes;
vector<Node *> active_nodes;
float ek_avg = -1;

std::default_random_engine random;

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
	Node *n0, *n1, *n2;
	n0 = new Node(float3(-1, 0, 0));
	n1 = new Node(float3(1, 0, 0));
	nodes.push_back(n0);
	nodes.push_back(n1);
	make_edge(n0, n1);
}

void step() {
	if (ek_avg < 0.5) {
		// one generation finished, prepare next one
		
		uniform_int_distribution<unsigned> bd(0, 1);

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
			if (bd(random)) continue;
			priority_queue<node_split_ptr> q2;
			for (Node *n : n0->edges) {
				q2.push(n);
			}
			Node *n1 = q2.top().get();
			assert(n1);
			uniform_real_distribution<float> lerp_dist(0, 1);
			Node *n2 = new Node(float3::lerp(n0->p, n1->p, lerp_dist(random)));
			n2->d = 0.5f * (n0->d + n1->d);
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
			if (bd(random)) continue;
			uniform_real_distribution<float> pd(-0.1, 0.1);
			Node *n2 = new Node(n0->p + float3(pd(random), pd(random), 0));
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
				f += float3(fd(random), fd(random), 0);
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

	win = createWindow().context(2, 1).visible(true).size(768, 768).title("Force Directed Terrain");
	win->makeContextCurrent();

	init();

	while (!win->shouldClose()) {
		glfwPollEvents();

		auto frame_start_time = chrono::steady_clock::now();
		while (chrono::steady_clock::now() - frame_start_time < chrono::milliseconds(15)) {
			step();
		}

		display(win->size());

		glFinish();
		win->swapBuffers();
	}

	delete win;
	glfwTerminate();

}