#include "skiplist.h"

#include <cassert>
#include <vector>

namespace skiplist {

struct skiplist_node
{
	skiplist_node *right = nullptr;
	skiplist_node *down  = nullptr;
	key_type key = 0;
	value_type val;

	skiplist_node(skiplist_node *r, skiplist_node *d, key_type k, const value_type &v)
		: right(r), down(d), key(k), val(v) {}
};

skiplist_type::skiplist_type(double p)
	: rand_level_(1.0 - p),
	  top_head_(new skiplist_node(nullptr, nullptr, {}, {}))
{
}

skiplist_type::~skiplist_type()
{
	auto head = top_head_;
	while (head != nullptr)
	{
		auto d = head->down;
		auto n = head;
		while (n != nullptr)
		{
			auto r = n->right;
			delete n;
			n = r;
		}
		head = d;
	}
	top_head_ = nullptr;
}

void skiplist_type::put(key_type key, const value_type &val)
{
	std::vector<skiplist_node*> query_path;
	auto n = this->top_head_;
	while (true)
	{
		while (n->right != nullptr && n->right->key < key)
			n = n->right;
		query_path.push_back(n);
		if (n->down == nullptr)
			break;
		n = n->down;
	}
	int level = rand_level_(rng_);
	skiplist_node *new_node = nullptr;
	while (level >= 0)
	{
		auto left = query_path.back();
		query_path.pop_back();
		new_node = new skiplist_node(left->right, new_node, key, val);
		left->right = new_node;
		--level;
		if (query_path.empty())
			break;
	}
	while (level >= 0)
	{
		new_node = new skiplist_node(nullptr, new_node, key, val);
		this->top_head_ = new skiplist_node(new_node, this->top_head_, {}, {});
		--level;
	}
}

std::string skiplist_type::get(key_type key) const
{
	auto n = top_head_;
	while (true)
	{
		while (n->right != nullptr && n->right->key < key)
			n = n->right;

		if (n->right != nullptr && n->right->key == key)
			return n->right->val;

		if (n->down == nullptr)
			break;
		n = n->down;
	}
	return "";
}

int skiplist_type::query_distance(key_type key) const
{
	int ret = 1;
	auto n = this->top_head_;

	while (true)
	{
		while (n->right != nullptr && n->right->key < key)
		{
			n = n->right;
			++ret;
		}

		if (n->right != nullptr && n->right->key == key)
			return ret + 1;

		if (n->down == nullptr)
			break;

		n = n->down;
		++ret;
	}
	return ret;
}

// ===== Stats computed by traversal =====

std::size_t skiplist_type::max_level() const
{
	std::size_t levels = 0;
	for (auto h = top_head_; h != nullptr; h = h->down)
		++levels;
	return levels;
}

double skiplist_type::average_height() const
{
	if (this->top_head_ == nullptr) return 0.0;

	// distinct keys = number of nodes in bottom level (excluding head)
	auto bottom = this->top_head_;
	while (bottom->down != nullptr) bottom = bottom->down;

	std::size_t bottom_nodes = 0;
	for (auto n = bottom->right; n != nullptr; n = n->right)
		++bottom_nodes;

	if (bottom_nodes == 0) return 0.0;

	// total nodes = sum of nodes across all levels (excluding heads)
	std::size_t total_nodes = 0;
	for (auto level_head = top_head_; level_head != nullptr; level_head = level_head->down)
	{
		for (auto n = level_head->right; n != nullptr; n = n->right)
			++total_nodes;
	}

	// average tower height = total nodes across all levels / number of distinct keys
	return static_cast<double>(total_nodes) / static_cast<double>(bottom_nodes);
}

} // namespace skiplist