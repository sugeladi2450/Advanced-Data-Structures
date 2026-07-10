#ifndef HW8_STACK_HPP
#define HW8_STACK_HPP

#include <atomic>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

template <class T>
class Stack {
public:
    void push(const T& value) {
        data_.push_back(value);
    }

    void push(T&& value) {
        data_.push_back(std::move(value));
    }

    bool pop(T& value) {
        if (data_.empty()) {
            return false;
        }
        value = std::move(data_.back());
        data_.pop_back();
        return true;
    }

    bool empty() const {
        return data_.empty();
    }

    std::size_t size() const {
        return data_.size();
    }

private:
    std::vector<T> data_;
};

template <class T>
class ThreadSafeStack {
public:
    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mu_);
        data_.push_back(value);
    }

    void push(T&& value) {
        std::lock_guard<std::mutex> lock(mu_);
        data_.push_back(std::move(value));
    }

    bool pop(T& value) {
        std::lock_guard<std::mutex> lock(mu_);
        if (data_.empty()) {
            return false;
        }
        value = std::move(data_.back());
        data_.pop_back();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mu_);
        return data_.empty();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return data_.size();
    }

private:
    mutable std::mutex mu_;
    std::vector<T> data_;
};

template <class T>
class LockFreeStack {
private:
    struct Node {
        explicit Node(const T& value) : data(value) {}
        explicit Node(T&& value) : data(std::move(value)) {}

        T data;
        Node* next = nullptr;
        Node* all_next = nullptr;
    };

public:
    LockFreeStack() = default;

    LockFreeStack(const LockFreeStack&) = delete;
    LockFreeStack& operator=(const LockFreeStack&) = delete;

    ~LockFreeStack() {
        Node* node = all_nodes_.load(std::memory_order_acquire);
        while (node != nullptr) {
            Node* next = node->all_next;
            delete node;
            node = next;
        }
    }

    void push(const T& value) {
        push_node(new Node(value));
    }

    void push(T&& value) {
        push_node(new Node(std::move(value)));
    }

    bool pop(T& value) {
        Node* old_head = head_.load(std::memory_order_acquire);
        while (old_head != nullptr &&
               !head_.compare_exchange_weak(old_head,
                                            old_head->next,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
        }

        if (old_head == nullptr) {
            return false;
        }

        value = std::move(old_head->data);
        return true;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == nullptr;
    }

    std::size_t size() const {
        std::size_t count = 0;
        for (Node* node = head_.load(std::memory_order_acquire);
             node != nullptr;
             node = node->next) {
            ++count;
        }
        return count;
    }

private:
    void remember_node(Node* node) {
        Node* old_head = all_nodes_.load(std::memory_order_relaxed);
        do {
            node->all_next = old_head;
        } while (!all_nodes_.compare_exchange_weak(old_head,
                                                  node,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed));
    }

    void push_node(Node* node) {
        remember_node(node);

        Node* old_head = head_.load(std::memory_order_relaxed);
        do {
            node->next = old_head;
        } while (!head_.compare_exchange_weak(old_head,
                                             node,
                                             std::memory_order_release,
                                             std::memory_order_relaxed));
    }

    std::atomic<Node*> head_{nullptr};
    std::atomic<Node*> all_nodes_{nullptr};
};

#endif
