#ifndef SKIPLIST_H
#define SKIPLIST_H

#include <cstdint>
#include <cstddef>
#include <random>
#include <string>

namespace skiplist {
using key_type = uint64_t;
using value_type = std::string;

struct skiplist_node;

class skiplist_type
{
private:
	// RNG for level generation
	std::mt19937_64 rng_;
	// geometric_distribution(k) with success prob (1-p): P(k)=p^k(1-p)
	std::geometric_distribution<int> rand_level_;

	skiplist_node *top_head_ = nullptr;

public:
	explicit skiplist_type(double p = 0.5);
	~skiplist_type();

	void put(key_type key, const value_type &val);
	std::string get(key_type key) const;

	// for hw1 only
	int query_distance(key_type key) const;

	std::size_t max_level() const;
	double average_height() const;
};

} // namespace skiplist

#endif // SKIPLIST_H