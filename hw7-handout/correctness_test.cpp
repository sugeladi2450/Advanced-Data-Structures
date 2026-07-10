#include "stack.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        return false;
    }
    return true;
}

bool check_exact_values(const std::vector<std::uint64_t>& values,
                        std::size_t expected_count,
                        const std::string& test_name) {
    if (!require(values.size() == expected_count,
                 test_name + ": wrong number of popped values")) {
        return false;
    }

    std::vector<unsigned char> seen(expected_count, 0);
    for (std::uint64_t value : values) {
        if (!require(value < expected_count, test_name + ": value out of range")) {
            return false;
        }
        if (!require(seen[value] == 0, test_name + ": duplicated value")) {
            return false;
        }
        seen[value] = 1;
    }

    return true;
}

bool test_sequential_behavior() {
    ThreadSafeStack<int> stack;
    if (!require(stack.empty(), "new stack should be empty")) {
        return false;
    }

    stack.push(1);
    stack.push(2);
    stack.push(3);

    if (!require(stack.size() == 3, "size after three pushes should be 3")) {
        return false;
    }

    int value = 0;
    if (!require(stack.pop(value) && value == 3, "first pop should return 3")) {
        return false;
    }
    if (!require(stack.pop(value) && value == 2, "second pop should return 2")) {
        return false;
    }
    if (!require(stack.pop(value) && value == 1, "third pop should return 1")) {
        return false;
    }

    return require(!stack.pop(value), "pop on empty stack should return false") &&
           require(stack.empty(), "stack should be empty after popping all values");
}

bool test_parallel_push() {
    constexpr unsigned thread_count = 8;
    constexpr std::size_t total_items = 200000;

    ThreadSafeStack<std::uint64_t> stack;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    for (unsigned tid = 0; tid < thread_count; ++tid) {
        workers.emplace_back([&, tid]() {
            const std::size_t begin = total_items * tid / thread_count;
            const std::size_t end = total_items * (tid + 1) / thread_count;
            for (std::size_t i = begin; i < end; ++i) {
                stack.push(i);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (!require(stack.size() == total_items, "parallel push: wrong final size")) {
        return false;
    }

    std::vector<std::uint64_t> values;
    values.reserve(total_items);
    std::uint64_t value = 0;
    while (stack.pop(value)) {
        values.push_back(value);
    }

    return check_exact_values(values, total_items, "parallel push");
}

bool test_parallel_pop() {
    constexpr unsigned thread_count = 8;
    constexpr std::size_t total_items = 200000;

    ThreadSafeStack<std::uint64_t> stack;
    for (std::size_t i = 0; i < total_items; ++i) {
        stack.push(i);
    }

    std::vector<std::thread> workers;
    std::vector<std::vector<std::uint64_t>> local_values(thread_count);
    std::atomic<bool> failed{false};
    workers.reserve(thread_count);

    for (unsigned tid = 0; tid < thread_count; ++tid) {
        workers.emplace_back([&, tid]() {
            const std::size_t begin = total_items * tid / thread_count;
            const std::size_t end = total_items * (tid + 1) / thread_count;
            local_values[tid].reserve(end - begin);

            for (std::size_t i = begin; i < end; ++i) {
                std::uint64_t value = 0;
                if (!stack.pop(value)) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                local_values[tid].push_back(value);
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    if (!require(!failed.load(std::memory_order_relaxed),
                 "parallel pop: pop returned false too early")) {
        return false;
    }
    if (!require(stack.empty(), "parallel pop: stack should be empty")) {
        return false;
    }

    std::vector<std::uint64_t> values;
    values.reserve(total_items);
    for (const auto& local : local_values) {
        values.insert(values.end(), local.begin(), local.end());
    }

    return check_exact_values(values, total_items, "parallel pop");
}

int main() {
    if (!test_sequential_behavior()) {
        return 1;
    }
    std::cout << "sequential behavior: passed\n";

    if (!test_parallel_push()) {
        return 1;
    }
    std::cout << "parallel push: passed\n";

    if (!test_parallel_pop()) {
        return 1;
    }
    std::cout << "parallel pop: passed\n";

    std::cout << "all correctness tests passed\n";
    return 0;
}
