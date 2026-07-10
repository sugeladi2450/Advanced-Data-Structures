#include "stack.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

struct BenchResult {
    double seconds = 0.0;
    std::uint64_t checksum = 0;
};

template <class F>
BenchResult time_it(F&& f) {
    const auto begin = Clock::now();
    const std::uint64_t checksum = f();
    const auto end = Clock::now();
    const std::chrono::duration<double> elapsed = end - begin;
    return {elapsed.count(), checksum};
}

std::size_t chunk_begin(std::size_t total, unsigned threads, unsigned tid) {
    return total * tid / threads;
}

std::size_t chunk_end(std::size_t total, unsigned threads, unsigned tid) {
    return total * (tid + 1) / threads;
}

BenchResult bench_serial_stack(std::size_t total_items) {
    return time_it([&]() {
        Stack<std::uint64_t> stack;
        for (std::size_t i = 0; i < total_items; ++i) {
            stack.push(i);
        }

        std::uint64_t sum = 0;
        std::uint64_t value = 0;
        while (stack.pop(value)) {
            sum += value;
        }
        return sum;
    });
}

template <class SharedStack>
BenchResult bench_parallel_stack(std::size_t total_items, unsigned threads) {
    return time_it([&]() {
        SharedStack stack;
        std::vector<std::thread> workers;
        workers.reserve(threads);

        for (unsigned tid = 0; tid < threads; ++tid) {
            workers.emplace_back([&, tid]() {
                const std::size_t begin = chunk_begin(total_items, threads, tid);
                const std::size_t end = chunk_end(total_items, threads, tid);
                for (std::size_t i = begin; i < end; ++i) {
                    stack.push(i);
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }
        workers.clear();
        std::vector<std::uint64_t> sums(threads, 0);
        for (unsigned tid = 0; tid < threads; ++tid) {
            workers.emplace_back([&, tid]() {
                const std::size_t begin = chunk_begin(total_items, threads, tid);
                const std::size_t end = chunk_end(total_items, threads, tid);
                std::uint64_t value = 0;
                for (std::size_t i = begin; i < end; ++i) {
                    if (stack.pop(value)) {
                        sums[tid] += value;
                    }
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }
        return std::accumulate(sums.begin(), sums.end(), std::uint64_t{0});
    });
}

double throughput_mops(const BenchResult& result, std::size_t total_items) {
    const double operations = static_cast<double>(total_items) * 2.0;
    return operations / result.seconds / 1e6;
}

void print_row(const std::string& name,
               unsigned threads,
               std::size_t total_items,
               const BenchResult& result,
               double serial_seconds) {
    std::cout << std::left << std::setw(30) << name
              << std::right << std::setw(8) << threads
              << std::setw(15) << total_items
              << std::setw(14) << std::fixed << std::setprecision(4) << result.seconds
              << std::setw(16) << std::fixed << std::setprecision(2)
              << throughput_mops(result, total_items)
              << std::setw(12) << std::fixed << std::setprecision(2)
              << serial_seconds / result.seconds
              << std::setw(18) << result.checksum << '\n';
}

int main(int argc, char* argv[]) {
    const std::size_t total_items = argc > 1 ? std::stoull(argv[1]) : 1000000;
    const unsigned max_threads = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 8;

    std::cout << "same total workload: push " << total_items
              << " + pop " << total_items << "\n\n";
    std::cout << std::left << std::setw(30) << "case"
              << std::right << std::setw(8) << "threads"
              << std::setw(15) << "items"
              << std::setw(14) << "seconds"
              << std::setw(16) << "M ops/s"
              << std::setw(12) << "speedup"
              << std::setw(18) << "checksum" << '\n';
    std::cout << std::string(113, '-') << '\n';

    const BenchResult serial = bench_serial_stack(total_items);
    print_row("Serial Stack", 1, total_items, serial, serial.seconds);

    for (unsigned threads = 2; threads <= max_threads; threads *= 2) {
        print_row("Parallel Mutex Stack",
                  threads,
                  total_items,
                  bench_parallel_stack<ThreadSafeStack<std::uint64_t>>(total_items, threads),
                  serial.seconds);
        print_row("Parallel LockFree Stack",
                  threads,
                  total_items,
                  bench_parallel_stack<LockFreeStack<std::uint64_t>>(total_items, threads),
                  serial.seconds);
    }

    return 0;
}
