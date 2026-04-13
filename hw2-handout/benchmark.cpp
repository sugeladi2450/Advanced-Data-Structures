#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rbtree/rbtree.h"
#include "splaytree/splaytree.h"

struct BenchmarkResult {
    double insert_ms = 0.0;
    double hit_search_ms = 0.0;
    double miss_search_ms = 0.0;
};

struct QueryCheck {
    std::size_t hit_count = 0;
    std::size_t miss_count = 0;
};

enum class WorkloadMode {
    RANDOM,
    LOCALITY,
    BOTH
};

struct BenchmarkConfig {
    std::size_t rounds = 3;
    int query_factor = 2;
    WorkloadMode mode = WorkloadMode::BOTH;
    std::vector<int> sizes;
};

struct WorkloadData {
    std::vector<int> insert_values;
    std::vector<int> hit_queries;
    std::vector<int> miss_queries;
    std::string description;
};

using Clock = std::chrono::steady_clock;

static bool isPositiveInteger(const std::string& text) {
    if (text.empty())
        return false;

    for (char ch : text) {
        if (ch < '0' || ch > '9')
            return false;
    }

    return true;
}

static bool isModeString(const std::string& text) {
    return text == "random" || text == "locality" || text == "both";
}

static WorkloadMode parseMode(const std::string& text) {
    if (text == "random")
        return WorkloadMode::RANDOM;
    if (text == "locality")
        return WorkloadMode::LOCALITY;
    if (text == "both")
        return WorkloadMode::BOTH;

    throw std::runtime_error("Unsupported mode: " + text);
}

static std::string modeName(WorkloadMode mode) {
    if (mode == WorkloadMode::RANDOM)
        return "random";
    if (mode == WorkloadMode::LOCALITY)
        return "locality";
    return "both";
}

static void printUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " [--rounds N] [--mode random|locality|both] [sizes...]\n"
        << "  " << program << " [rounds] [mode] [sizes...]\n"
        << '\n'
        << "Examples:\n"
        << "  " << program << '\n'
        << "  " << program << " --mode locality 20000 50000\n"
        << "  " << program << " 5 both 10000 50000 100000\n";
}

template <typename Tree>
QueryCheck runQueries(
    Tree& tree,
    const std::vector<int>& hit_queries,
    const std::vector<int>& miss_queries,
    double& hit_search_ms,
    double& miss_search_ms
) {
    std::size_t hit_count = 0;
    auto start = Clock::now();
    for (int value : hit_queries)
        hit_count += tree.search(value) ? 1U : 0U;
    auto end = Clock::now();
    hit_search_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::size_t miss_count = 0;
    start = Clock::now();
    for (int value : miss_queries)
        miss_count += tree.search(value) ? 1U : 0U;
    end = Clock::now();
    miss_search_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {hit_count, miss_count};
}

template <typename Tree>
BenchmarkResult runSingleRound(const WorkloadData& workload) {
    Tree tree;

    auto start = Clock::now();
    for (int value : workload.insert_values)
        tree.insert(value);
    auto end = Clock::now();

    BenchmarkResult result;
    result.insert_ms = std::chrono::duration<double, std::milli>(end - start).count();

    QueryCheck check = runQueries(
        tree,
        workload.hit_queries,
        workload.miss_queries,
        result.hit_search_ms,
        result.miss_search_ms
    );

    if (check.hit_count != workload.hit_queries.size())
        throw std::runtime_error("Successful search benchmark returned missing values.");
    if (check.miss_count != 0)
        throw std::runtime_error("Unsuccessful search benchmark unexpectedly found values.");

    return result;
}

static std::vector<int> makeEvenValues(int size) {
    std::vector<int> values(size);
    for (int i = 0; i < size; ++i)
        values[i] = 2 * (i + 1);
    return values;
}

static std::vector<int> makeMissPool(int size) {
    std::vector<int> values(size + 1);
    for (int i = 0; i <= size; ++i)
        values[i] = 2 * i + 1;
    return values;
}

static std::vector<int> makeUniformQueriesFromPool(
    const std::vector<int>& pool,
    int count,
    std::mt19937& rng
) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pool.size()) - 1);
    std::vector<int> queries;
    queries.reserve(count);

    for (int i = 0; i < count; ++i)
        queries.push_back(pool[dist(rng)]);

    return queries;
}

static std::vector<int> makeRandomInsertValues(int size, std::mt19937& rng) {
    std::vector<int> values = makeEvenValues(size);
    std::shuffle(values.begin(), values.end(), rng);
    return values;
}

static std::vector<int> makeLocalityInsertValues(int size, std::mt19937& rng) {
    std::vector<int> sorted_values = makeEvenValues(size);
    const int block_size = std::clamp(size / 20, 32, 512);
    const int block_count = (size + block_size - 1) / block_size;

    std::vector<int> block_ids(block_count);
    std::iota(block_ids.begin(), block_ids.end(), 0);
    std::shuffle(block_ids.begin(), block_ids.end(), rng);

    std::bernoulli_distribution reverse_block(0.5);
    std::vector<int> values;
    values.reserve(size);

    for (int block_id : block_ids) {
        int start = block_id * block_size;
        int end = std::min(size, start + block_size);

        if (reverse_block(rng)) {
            for (int i = end - 1; i >= start; --i)
                values.push_back(sorted_values[i]);
        } else {
            for (int i = start; i < end; ++i)
                values.push_back(sorted_values[i]);
        }
    }

    return values;
}

static std::vector<int> makeHotPool(const std::vector<int>& insert_values, std::mt19937& rng) {
    std::vector<int> sorted_values = insert_values;
    std::sort(sorted_values.begin(), sorted_values.end());

    int hot_count = std::clamp(static_cast<int>(sorted_values.size() / 2000), 4, 8);
    hot_count = std::min(hot_count, static_cast<int>(sorted_values.size()));
    if (hot_count == 0)
        return sorted_values;

    const int window_count = std::min(2, hot_count);
    const int window_size = std::max(1, hot_count / window_count);
    std::uniform_int_distribution<int> start_dist(
        0,
        std::max(0, static_cast<int>(sorted_values.size()) - window_size)
    );

    std::vector<int> hot_pool;
    hot_pool.reserve(hot_count);

    for (int i = 0; i < window_count && static_cast<int>(hot_pool.size()) < hot_count; ++i) {
        int start = start_dist(rng);
        int end = std::min(static_cast<int>(sorted_values.size()), start + window_size);
        for (int j = start; j < end && static_cast<int>(hot_pool.size()) < hot_count; ++j)
            hot_pool.push_back(sorted_values[j]);
    }

    for (int i = 0; i < static_cast<int>(sorted_values.size()) &&
                    static_cast<int>(hot_pool.size()) < hot_count; ++i) {
        hot_pool.push_back(sorted_values[i]);
    }

    return hot_pool;
}

static std::vector<int> makeHotMissPool(const std::vector<int>& hot_hit_pool) {
    std::vector<int> hot_miss_pool;
    hot_miss_pool.reserve(hot_hit_pool.size() * 2);

    for (int value : hot_hit_pool) {
        hot_miss_pool.push_back(value - 1);
        hot_miss_pool.push_back(value + 1);
    }

    return hot_miss_pool;
}

static std::vector<int> makeLocalityQueries(
    const std::vector<int>& hot_pool,
    const std::vector<int>& full_pool,
    int count,
    std::mt19937& rng
) {
    if (hot_pool.empty())
        return makeUniformQueriesFromPool(full_pool, count, rng);

    std::vector<int> queries;
    queries.reserve(count);

    std::bernoulli_distribution use_anchor_key(0.96);
    std::bernoulli_distribution use_hot_phase(0.995);
    std::uniform_int_distribution<int> hot_dist(0, static_cast<int>(hot_pool.size()) - 1);
    std::uniform_int_distribution<int> full_dist(0, static_cast<int>(full_pool.size()) - 1);
    std::uniform_int_distribution<int> burst_len_dist(64, 256);

    while (static_cast<int>(queries.size()) < count) {
        bool choose_hot = use_hot_phase(rng);
        int remaining = count - static_cast<int>(queries.size());
        int burst_length = std::min(remaining, burst_len_dist(rng));

        if (choose_hot) {
            int anchor = hot_dist(rng);
            int left = std::max(0, anchor - 1);
            int right = std::min(static_cast<int>(hot_pool.size()) - 1, anchor + 1);
            std::uniform_int_distribution<int> local_dist(left, right);

            for (int i = 0; i < burst_length; ++i) {
                if (use_anchor_key(rng))
                    queries.push_back(hot_pool[anchor]);
                else
                    queries.push_back(hot_pool[local_dist(rng)]);
            }
        } else {
            for (int i = 0; i < burst_length; ++i)
                queries.push_back(full_pool[full_dist(rng)]);
        }
    }

    return queries;
}

static WorkloadData makeWorkloadData(
    int size,
    int query_factor,
    WorkloadMode mode,
    std::mt19937& rng
) {
    const int query_count = size * query_factor;

    if (mode == WorkloadMode::RANDOM) {
        std::vector<int> insert_values = makeRandomInsertValues(size, rng);
        std::vector<int> miss_pool = makeMissPool(size);

        return {
            insert_values,
            makeUniformQueriesFromPool(insert_values, query_count, rng),
            makeUniformQueriesFromPool(miss_pool, query_count, rng),
            "globally shuffled inserts; uniform hit and miss queries"
        };
    }

    std::vector<int> insert_values = makeLocalityInsertValues(size, rng);
    std::vector<int> hot_hit_pool = makeHotPool(insert_values, rng);
    std::vector<int> miss_pool = makeMissPool(size);
    std::vector<int> hot_miss_pool = makeHotMissPool(hot_hit_pool);

    return {
        insert_values,
        makeLocalityQueries(hot_hit_pool, insert_values, query_count, rng),
        makeLocalityQueries(hot_miss_pool, miss_pool, query_count, rng),
        "clustered inserts; very small hot set with long repeated bursts to emphasize temporal locality"
    };
}

static BenchmarkConfig parseArgs(int argc, char* argv[]) {
    BenchmarkConfig config;
    bool rounds_set = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        }

        if (arg == "--rounds" || arg == "-r") {
            if (i + 1 >= argc || !isPositiveInteger(argv[i + 1]))
                throw std::runtime_error("Expected a positive integer after --rounds.");
            config.rounds = std::stoul(argv[++i]);
            rounds_set = true;
            continue;
        }

        if (arg == "--mode" || arg == "-m") {
            if (i + 1 >= argc || !isModeString(argv[i + 1]))
                throw std::runtime_error("Expected random, locality, or both after --mode.");
            config.mode = parseMode(argv[++i]);
            continue;
        }

        if (isModeString(arg)) {
            config.mode = parseMode(arg);
            continue;
        }

        if (!rounds_set && isPositiveInteger(arg) && config.sizes.empty() && i + 1 < argc) {
            config.rounds = std::stoul(arg);
            rounds_set = true;
            continue;
        }

        if (isPositiveInteger(arg)) {
            int size = std::stoi(arg);
            if (size <= 0)
                throw std::runtime_error("Tree sizes must be positive.");
            config.sizes.push_back(size);
            continue;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    if (config.rounds == 0)
        throw std::runtime_error("Rounds must be positive.");
    if (config.sizes.empty())
        config.sizes = {10000, 50000, 100000};

    return config;
}

static void printHeader(const BenchmarkConfig& config, const WorkloadData& workload, const std::string& mode) {
    std::cout << "\nWorkload: " << mode << '\n';
    std::cout << "Description: " << workload.description << '\n';
    std::cout << "Rounds: " << config.rounds << '\n';
    std::cout << "Insert(ms) = total time for N insertions\n";
    std::cout << "Search Hit(ms) = total time for N * " << config.query_factor
              << " successful lookups\n";
    std::cout << "Search Miss(ms) = total time for N * " << config.query_factor
              << " unsuccessful lookups\n";
    std::cout << std::left
              << std::setw(10) << "N"
              << std::setw(18) << "Tree"
              << std::setw(16) << "Insert(ms)"
              << std::setw(20) << "Search Hit(ms)"
              << std::setw(20) << "Search Miss(ms)"
              << '\n';
    std::cout << std::string(84, '-') << '\n';
}

static void printRow(int size, const std::string& name, const BenchmarkResult& result) {
    std::cout << std::left
              << std::setw(10) << size
              << std::setw(18) << name
              << std::setw(16) << std::fixed << std::setprecision(3) << result.insert_ms
              << std::setw(20) << std::fixed << std::setprecision(3) << result.hit_search_ms
              << std::setw(20) << std::fixed << std::setprecision(3) << result.miss_search_ms
              << '\n';
}

static void runBenchmarkForMode(const BenchmarkConfig& config, WorkloadMode mode) {
    std::mt19937 header_rng(20260331);
    WorkloadData example_workload = makeWorkloadData(1000, config.query_factor, mode, header_rng);
    printHeader(config, example_workload, modeName(mode));

    for (int size : config.sizes) {
        BenchmarkResult rb_total;
        BenchmarkResult splay_total;

        for (std::size_t round = 0; round < config.rounds; ++round) {
            std::mt19937 rng(static_cast<std::mt19937::result_type>(20260331 + size * 17 + round));
            WorkloadData workload = makeWorkloadData(size, config.query_factor, mode, rng);

            BenchmarkResult rb_result = runSingleRound<RedBlackTree>(workload);
            BenchmarkResult splay_result = runSingleRound<SplayTree>(workload);

            rb_total.insert_ms += rb_result.insert_ms;
            rb_total.hit_search_ms += rb_result.hit_search_ms;
            rb_total.miss_search_ms += rb_result.miss_search_ms;

            splay_total.insert_ms += splay_result.insert_ms;
            splay_total.hit_search_ms += splay_result.hit_search_ms;
            splay_total.miss_search_ms += splay_result.miss_search_ms;
        }

        rb_total.insert_ms /= config.rounds;
        rb_total.hit_search_ms /= config.rounds;
        rb_total.miss_search_ms /= config.rounds;

        splay_total.insert_ms /= config.rounds;
        splay_total.hit_search_ms /= config.rounds;
        splay_total.miss_search_ms /= config.rounds;

        printRow(size, "RedBlackTree", rb_total);
        printRow(size, "SplayTree", splay_total);
    }
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config = parseArgs(argc, argv);

    if (config.mode == WorkloadMode::BOTH) {
        runBenchmarkForMode(config, WorkloadMode::RANDOM);
        runBenchmarkForMode(config, WorkloadMode::LOCALITY);
    } else {
        runBenchmarkForMode(config, config.mode);
    }

    return 0;
}
