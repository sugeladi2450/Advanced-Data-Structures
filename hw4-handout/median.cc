#include <vector>
#include <span>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
#include <numeric>
#include <iostream>

using span = std::span<int>;
using span_it = span::iterator;

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

int linear_nth(span s, int nth, int Q, int depth = 1, int* max_depth = nullptr)
{
    if (max_depth)
        *max_depth = std::max(*max_depth, depth);

    assert(nth < std::ssize(s));
    assert(nth >= 0);
    if (std::ssize(s) <= Q)
        return trivial_nth(s, nth);

    std::vector<int> mids;
    {
        int i = 0;
        for (; i + Q < std::ssize(s); i += Q)
            mids.push_back(trivial_median({s.begin() + i, s.begin() + i + Q}));
        mids.push_back(trivial_median({s.begin() + i, s.end()}));
    }

    int pivot = linear_nth(mids, mids.size() / 2, Q, depth + 1, max_depth);
    auto [less_it, greater_it] = partition(s, pivot);
    int less_cnt = static_cast<int>(less_it - s.begin());
    if (nth < less_cnt)
        return linear_nth({s.begin(), less_it}, nth, Q, depth + 1, max_depth);
    int le_cnt = static_cast<int>(greater_it - s.begin());
    if (nth < le_cnt)
        return pivot;
    return linear_nth({greater_it, s.end()}, nth - le_cnt, Q, depth + 1, max_depth);
}

int quick_select(span s, int nth, int Q)
{
    assert(nth < std::ssize(s));
    assert(nth >= 0);
    if (std::ssize(s) <= Q)
        return trivial_nth(s, nth);

    int pivot = s[0];
    auto [less_it, greater_it] = partition(s, pivot);
    int less_cnt = static_cast<int>(less_it - s.begin());
    if (nth < less_cnt)
        return quick_select({s.begin(), less_it}, nth, Q);
    int le_cnt = static_cast<int>(greater_it - s.begin());
    if (nth < le_cnt)
        return pivot;
    return quick_select({greater_it, s.end()}, nth - le_cnt, Q);
}

using myclock = std::chrono::system_clock;

myclock::duration test_linear(span s, int nth, int Q)
{
    std::vector<int> copy(s.begin(), s.end());
    auto t1 = myclock::now();
    linear_nth(copy, nth, Q);
    auto t2 = myclock::now();
    return t2 - t1;
}

myclock::duration test_quick(span s, int nth, int Q)
{
    std::vector<int> copy(s.begin(), s.end());
    auto t1 = myclock::now();
    quick_select(copy, nth, Q);
    auto t2 = myclock::now();
    return t2 - t1;
}

int test_linear_depth(span s, int nth, int Q)
{
    std::vector<int> copy(s.begin(), s.end());
    int max_depth = 0;
    linear_nth(copy, nth, Q, 1, &max_depth);
    return max_depth;
}

template <class T>
T p90(std::vector<T>& v)
{
    std::ranges::sort(v);
    return v[v.size() * 90 / 100];
}

void test_with_N(int N)
{
    std::mt19937_64 rng;
    std::vector<int> input_seq(N);
    for (int i = 0; i < N; ++i)
        input_seq[i] = i;

    auto input_rand = input_seq;
    std::ranges::shuffle(input_rand, rng);
    std::uniform_int_distribution<int> rand_nth(0, N - 1);

    std::cout << "N=" << N << ":\n";
    std::cout << "Q|seq_linear_t|rand_linear_t|seq_quick_t|rand_quick_t|seq_linear_depth|rand_linear_depth\n";

    for (int Q : {5, 6, 7, 8, 9, 10, 11})
    {
        std::vector<myclock::duration> time_seq_linear, time_rand_linear;
        std::vector<myclock::duration> time_seq_quick, time_rand_quick;
        std::vector<int> depth_seq_linear, depth_rand_linear;

        for (int times = 0; times < 1000; ++times)
        {
            int nth = rand_nth(rng);

            time_seq_linear.push_back(test_linear(input_seq, nth, Q));
            time_rand_linear.push_back(test_linear(input_rand, nth, Q));
            time_seq_quick.push_back(test_quick(input_seq, nth, Q));
            time_rand_quick.push_back(test_quick(input_rand, nth, Q));

            depth_seq_linear.push_back(test_linear_depth(input_seq, nth, Q));
            depth_rand_linear.push_back(test_linear_depth(input_rand, nth, Q));
        }

        std::cout << Q 
                  << "|" << p90(time_seq_linear)
                  << "|" << p90(time_rand_linear)
                  << "|" << p90(time_seq_quick)
                  << "|" << p90(time_rand_quick)
                  << "|" << p90(depth_seq_linear)
                  << "|" << p90(depth_rand_linear)
                  << "\n";
    }
}

int main()
{
    for (int N : {100, 1000, 2000})
        test_with_N(N);
}