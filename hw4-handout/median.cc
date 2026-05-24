#include <vector>
#include <span>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>

using span = std::span<int>;
using span_it = span::iterator;
using myclock = std::chrono::high_resolution_clock;

std::pair<span_it, span_it> partition(span s, int pivot)
{
    std::vector<int> output_buffer(s.size());
    auto less_p = output_buffer.begin(), greater_p = output_buffer.end();
    for (int x : s)
    {
        if (x < pivot)
            *(less_p++) = x;
        else if (x > pivot)
            *(--greater_p) = x;
    }
    auto t1 = std::copy(output_buffer.begin(), less_p, s.begin());
    auto t2 = t1 + (greater_p - less_p);
    std::fill(t1, t2, pivot);
    std::copy(greater_p, output_buffer.end(), t2);
    return {t1, t2};
}

int trivial_nth(span s, int nth)
{
    auto nth_it = s.begin() + nth;
    std::ranges::nth_element(s, s.begin() + nth);
    return *nth_it;
}

int trivial_median(span s)
{
    return trivial_nth(s, s.size() / 2);
}

struct LinearResult {
    int value;
    int max_depth;
    int call_count;
};

LinearResult linear_nth_impl(span s, int nth, int Q, int depth, int& max_depth, int& call_count)
{
    ++call_count;
    max_depth = std::max(max_depth, depth);

    assert(nth < std::ssize(s));
    assert(nth >= 0);
    if (std::ssize(s) <= Q)
        return {trivial_nth(s, nth), max_depth, call_count};

    std::vector<int> mids;
    {
        int i = 0;
        for (; i + Q < std::ssize(s); i += Q)
            mids.push_back(trivial_median({s.begin() + i, s.begin() + i + Q}));
        mids.push_back(trivial_median({s.begin() + i, s.end()}));
    }

    int pivot = linear_nth_impl(mids, mids.size() / 2, Q, depth + 1, max_depth, call_count).value;
    auto [less_it, greater_it] = partition(s, pivot);
    int less_cnt = static_cast<int>(less_it - s.begin());
    if (nth < less_cnt)
        return linear_nth_impl({s.begin(), less_it}, nth, Q, depth + 1, max_depth, call_count);
    int le_cnt = static_cast<int>(greater_it - s.begin());
    if (nth < le_cnt)
        return {pivot, max_depth, call_count};
    return linear_nth_impl({greater_it, s.end()}, nth - le_cnt, Q, depth + 1, max_depth, call_count);
}

inline int linear_nth(span s, int nth, int Q, int* out_depth = nullptr, int* out_calls = nullptr)
{
    int max_depth = 0, call_count = 0;
    auto result = linear_nth_impl(s, nth, Q, 1, max_depth, call_count);
    if (out_depth) *out_depth = max_depth;
    if (out_calls) *out_calls = call_count;
    return result.value;
}

int quick_select(span s, int nth, std::mt19937_64& rng)
{
    assert(nth < std::ssize(s));
    assert(nth >= 0);
    if (s.size() <= 1)
        return s[0];

    std::uniform_int_distribution<int> dist(0, static_cast<int>(s.size()) - 1);
    int pivot = s[dist(rng)];
    auto [less_it, greater_it] = partition(s, pivot);
    int less_cnt = static_cast<int>(less_it - s.begin());
    if (nth < less_cnt)
        return quick_select({s.begin(), less_it}, nth, rng);
    int le_cnt = static_cast<int>(greater_it - s.begin());
    if (nth < le_cnt)
        return pivot;
    return quick_select({greater_it, s.end()}, nth - le_cnt, rng);
}

inline double to_us(myclock::duration d)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

struct Stats {
    double avg, p50, p90, p99, max;
};

Stats compute_stats(std::vector<double>& v)
{
    std::ranges::sort(v);
    size_t n = v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return {
        sum / n,
        v[n * 50 / 100],
        v[n * 90 / 100],
        v[n * 99 / 100],
        v.back()
    };
}

enum class DistType { SEQUENTIAL, RANDOM, REPEATED, NEARLY_SORTED };

std::vector<int> generate_data(int N, DistType type, std::mt19937_64& rng)
{
    std::vector<int> data(N);
    switch (type) {
    case DistType::SEQUENTIAL:
        std::iota(data.begin(), data.end(), 0);
        break;
    case DistType::RANDOM:
        std::iota(data.begin(), data.end(), 0);
        std::ranges::shuffle(data, rng);
        break;
    case DistType::REPEATED:
        for (int i = 0; i < N; ++i)
            data[i] = i % (N / 10);
        std::ranges::shuffle(data, rng);
        break;
    case DistType::NEARLY_SORTED:
        std::iota(data.begin(), data.end(), 0);
        for (int i = 0; i < N / 20; ++i)
            std::swap(data[std::uniform_int_distribution<int>(0, N-1)(rng)],
                      data[std::uniform_int_distribution<int>(0, N-1)(rng)]);
        break;
    }
    return data;
}

std::string dist_name(DistType t)
{
    switch (t) {
    case DistType::SEQUENTIAL: return "顺序";
    case DistType::RANDOM: return "随机";
    case DistType::REPEATED: return "重复";
    case DistType::NEARLY_SORTED: return "近似有序";
    }
    return "";
}

void run_single_test(int N, DistType dtype, int Q, int trials,
                     std::mt19937_64& rng,
                     std::vector<double>& out_linear_time,
                     std::vector<int>& out_linear_depth,
                     std::vector<int>& out_linear_calls,
                     std::vector<double>& out_quick_time)
{
    auto data = generate_data(N, dtype, rng);
    std::uniform_int_distribution<int> rand_nth(0, N - 1);

    for (int t = 0; t < trials; ++t) {
        int nth = rand_nth(rng);

        {
            std::vector<int> copy = data;
            int depth = 0, calls = 0;
            auto t1 = myclock::now();
            linear_nth(copy, nth, Q, &depth, &calls);
            auto t2 = myclock::now();
            out_linear_time.push_back(to_us(t2 - t1));
            out_linear_depth.push_back(depth);
            out_linear_calls.push_back(calls);
        }

        {
            std::vector<int> copy = data;
            auto t1 = myclock::now();
            quick_select(copy, nth, rng);
            auto t2 = myclock::now();
            out_quick_time.push_back(to_us(t2 - t1));
        }
    }
}

void experiment_Q_effect(int N, const std::vector<int>& Q_values, int trials)
{
    std::mt19937_64 rng(42);
    std::cout << "\n==================== 实验1：Q 值对算法性能的影响 (N=" << N << ") ====================\n\n";

    std::cout << std::left << std::setw(6) << "Q"
              << "| " << std::setw(12) << "线性-平均(μs)"
              << "| " << std::setw(12) << "线性-P90(μs)"
              << "| " << std::setw(12) << "线性-P99(μs)"
              << "| " << std::setw(10) << "快速-P90(μs)"
              << "| " << std::setw(10) << "平均深度"
              << "| " << std::setw(10) << "最大深度"
              << "| " << std::setw(10) << "平均调用数"
              << "| " << std::setw(10) << "最大调用数"
              << "|\n";
    std::cout << std::string(100, '-') << "\n";

    for (int Q : Q_values) {
        std::vector<double> linear_time, quick_time;
        std::vector<int> depths, calls;

        run_single_test(N, DistType::RANDOM, Q, trials, rng,
                        linear_time, depths, calls, quick_time);

        auto lt = compute_stats(linear_time);
        auto qt = compute_stats(quick_time);
        std::ranges::sort(depths);
        std::ranges::sort(calls);
        double avg_depth = std::accumulate(depths.begin(), depths.end(), 0.0) / depths.size();

        std::cout << std::left << std::setw(6) << Q
                  << "| " << std::setw(12) << std::fixed << std::setprecision(2) << lt.avg
                  << "| " << std::setw(12) << lt.p90
                  << "| " << std::setw(12) << lt.p99
                  << "| " << std::setw(10) << qt.p90
                  << "| " << std::setw(10) << std::setprecision(2) << avg_depth
                  << "| " << std::setw(10) << depths.back()
                  << "| " << std::setw(10) << std::setprecision(2) << (std::accumulate(calls.begin(), calls.end(), 0.0) / calls.size())
                  << "| " << std::setw(10) << calls.back()
                  << "|\n";
    }
}

void experiment_data_distribution(int N, int Q, int trials)
{
    std::mt19937_64 rng(42);
    std::cout << "\n==================== 实验2：不同数据分布下的表现 (N=" << N << ", Q=" << Q << ") ====================\n\n";

    std::cout << std::left << std::setw(14) << "数据分布"
              << "| " << std::setw(12) << "线性-P90(μs)"
              << "| " << std::setw(12) << "快速-P90(μs)"
              << "| " << std::setw(10) << "平均深度"
              << "| " << std::setw(10) << "最大深度"
              << "| " << std::setw(10) << "平均调用数"
              << "|\n";
    std::cout << std::string(80, '-') << "\n";

    for (auto dtype : {DistType::RANDOM, DistType::SEQUENTIAL, DistType::REPEATED, DistType::NEARLY_SORTED}) {
        std::vector<double> linear_time, quick_time;
        std::vector<int> depths, calls;

        run_single_test(N, dtype, Q, trials, rng,
                        linear_time, depths, calls, quick_time);

        auto lt = compute_stats(linear_time);
        auto qt = compute_stats(quick_time);
        double avg_depth = std::accumulate(depths.begin(), depths.end(), 0.0) / depths.size();
        double avg_calls = std::accumulate(calls.begin(), calls.end(), 0.0) / calls.size();

        std::cout << std::left << std::setw(14) << dist_name(dtype)
                  << "| " << std::setw(12) << std::fixed << std::setprecision(2) << lt.p90
                  << "| " << std::setw(12) << qt.p90
                  << "| " << std::setw(10) << std::setprecision(2) << avg_depth
                  << "| " << std::setw(10) << depths.back()
                  << "| " << std::setw(10) << std::setprecision(2) << avg_calls
                  << "|\n";
    }
}

void experiment_scale_effect(const std::vector<int>& N_values, const std::vector<int>& Q_values, int trials)
{
    std::mt19937_64 rng(42);
    std::cout << "\n==================== 实验3：不同规模下 Q 对时间的影响 ====================\n\n";

    std::cout << std::left << std::setw(8) << "N \\ Q";
    for (int Q : Q_values)
        std::cout << "| " << std::setw(10) << Q;
    std::cout << "| " << std::setw(12) << "Quick-P90" << "|\n";
    std::cout << std::string(12 + 12 * Q_values.size() + 14, '-') << "\n";

    for (int N : N_values) {
        std::cout << std::left << std::setw(8) << N;
        std::vector<double> all_quick;
        for (int Q : Q_values) {
            std::vector<double> linear_time, quick_time;
            std::vector<int> depths, calls;
            run_single_test(N, DistType::RANDOM, Q, trials, rng,
                            linear_time, depths, calls, quick_time);
            auto lt = compute_stats(linear_time);
            std::cout << "| " << std::setw(10) << std::fixed << std::setprecision(1) << lt.p90;
            all_quick.insert(all_quick.end(), quick_time.begin(), quick_time.end());
        }
        auto qs = compute_stats(all_quick);
        std::cout << "| " << std::setw(10) << std::fixed << std::setprecision(1) << qs.p90 << "|\n";
    }
}

void experiment_depth_analysis(const std::vector<int>& N_values, const std::vector<int>& Q_values, int trials)
{
    std::mt19937_64 rng(42);
    std::cout << "\n==================== 实验4：Q 对递归深度和调用次数的影响 ====================\n\n";

    for (int N : N_values) {
        std::cout << "--- N=" << N << " ---\n";
        std::cout << std::left << std::setw(6) << "Q"
                  << "| " << std::setw(12) << "平均深度"
                  << "| " << std::setw(12) << "P90深度"
                  << "| " << std::setw(12) << "最大深度"
                  << "| " << std::setw(12) << "平均调用数"
                  << "| " << std::setw(12) << "P90调用数"
                  << "| " << std::setw(12) << "最大调用数"
                  << "|\n";
        std::cout << std::string(80, '-') << "\n";

        for (int Q : Q_values) {
            std::vector<double> linear_time, quick_time;
            std::vector<int> depths, calls;
            run_single_test(N, DistType::RANDOM, Q, trials, rng,
                            linear_time, depths, calls, quick_time);

            std::ranges::sort(depths);
            std::ranges::sort(calls);
            double avg_depth = std::accumulate(depths.begin(), depths.end(), 0.0) / depths.size();
            double avg_calls = std::accumulate(calls.begin(), calls.end(), 0.0) / calls.size();

            std::cout << std::left << std::setw(6) << Q
                      << "| " << std::setw(12) << std::fixed << std::setprecision(2) << avg_depth
                      << "| " << std::setw(12) << depths[depths.size() * 90 / 100]
                      << "| " << std::setw(12) << depths.back()
                      << "| " << std::setw(12) << std::setprecision(2) << avg_calls
                      << "| " << std::setw(12) << calls[calls.size() * 90 / 100]
                      << "| " << std::setw(12) << calls.back()
                      << "|\n";
        }
        std::cout << "\n";
    }
}

int main()
{
    std::vector<int> Q_values = {3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 75, 101};
    std::vector<int> N_small = {100, 500, 1000};
    std::vector<int> N_medium = {5000, 10000, 20000};

    int trials_fast = 500;
    int trials_full = 200;

    experiment_Q_effect(5000, Q_values, trials_full);

    experiment_scale_effect(N_medium, Q_values, trials_full);

    experiment_depth_analysis(N_medium, Q_values, trials_full);

    experiment_data_distribution(5000, 5, trials_full);
    experiment_data_distribution(5000, 21, trials_full);

    experiment_depth_analysis(N_small, {3, 5, 7, 9, 11, 15, 21, 31}, trials_fast);

    return 0;
}