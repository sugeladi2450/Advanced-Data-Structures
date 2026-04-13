#include "rbtree.h"
#include <iostream>

// The color of new node is always red
Node::Node(int data) : data(data), color(RED), parent(nullptr), left(nullptr), right(nullptr) {} 

RedBlackTree::RedBlackTree() : root(nullptr) {}

RedBlackTree::~RedBlackTree() {
    clear(root);
}

void RedBlackTree::rotateLeft(Node* x) {
    // Left rotation logic
    Node* y = x->right;
    x->right = y->left;
    if (y->left != nullptr)
        y->left->parent = x;
    y->parent = x->parent;
    if (x->parent == nullptr)
        root = y;
    else if (x == x->parent->left)
        x->parent->left = y;
    else
        x->parent->right = y;
    y->left = x;
    x->parent = y;
}

void RedBlackTree::rotateRight(Node* x) {
    // Right rotation logic
    Node* y = x->left;
    x->left = y->right;
    if (y->right != nullptr)
        y->right->parent = x;
    y->parent = x->parent;
    if (x->parent == nullptr)
        root = y;
    else if (x == x->parent->right)
        x->parent->right = y;
    else
        x->parent->left = y;
    y->right = x;
    x->parent = y;
}

void RedBlackTree::fixViolation(Node* pt) {
    // Fix violation logic
    Node* parent_pt = nullptr;
    Node* grand_parent_pt = nullptr;

    
    while ((pt != root) && (pt->color == RED) && (pt->parent->color == RED)) {
        parent_pt = pt->parent;
        grand_parent_pt = pt->parent->parent;

        if (parent_pt == grand_parent_pt->left) {
            Node* uncle_pt = grand_parent_pt->right;

            // Uncle is red
            if (uncle_pt != nullptr && uncle_pt->color == RED) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            } else {// Uncle is black
                if (pt == parent_pt->right) {
                    rotateLeft(parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                rotateRight(grand_parent_pt);
                parent_pt->color = BLACK;
                grand_parent_pt->color = RED;
                pt = parent_pt;
            }
        } else {
            Node* uncle_pt = grand_parent_pt->left;

            if ((uncle_pt != nullptr) && (uncle_pt->color == RED)) {
                grand_parent_pt->color = RED;
                parent_pt->color = BLACK;
                uncle_pt->color = BLACK;
                pt = grand_parent_pt;
            } else {
                if (pt == parent_pt->left) {
                    rotateRight(parent_pt);
                    pt = parent_pt;
                    parent_pt = pt->parent;
                }

                rotateLeft(grand_parent_pt);
                parent_pt->color = BLACK;
                grand_parent_pt->color = RED;
                pt = parent_pt;
            }
        }
    }

    root->color = BLACK;
}

Node* RedBlackTree::BSTInsert(Node* root, Node* pt) {
    // Binary search tree insert logic
    if (root == nullptr)
        return pt;

    if (pt->data < root->data) {
        root->left = BSTInsert(root->left, pt);
        root->left->parent = root;
    } else if (pt->data > root->data) {
        root->right = BSTInsert(root->right, pt);
        root->right->parent = root;
    }

    return root;
}

void RedBlackTree::inorderUtil(Node* root) {
    // Inorder traversal logic
    if (root == nullptr)
        return;
    inorderUtil(root->left);
    std::cout << root->data << "(" << (root->color == RED ? "R" : "B") << ") ";
    inorderUtil(root->right);
}

Node* RedBlackTree::searchNode(Node* root, int data) const {
    if (root == nullptr || root->data == data)
        return root;

    if (data < root->data)
        return searchNode(root->left, data);

    return searchNode(root->right, data);
}

void RedBlackTree::clear(Node* root) {
    if (root == nullptr)
        return;

    clear(root->left);
    clear(root->right);
    delete root;
}

void RedBlackTree::insert(const int data) {
    // Insertion logic
    Node* new_node = new Node(data);
    root = BSTInsert(root, new_node);

    if (new_node != root && new_node->parent == nullptr) {
        delete new_node;
        return;
    }

    fixViolation(new_node);
}

bool RedBlackTree::search(int data) const {
    return searchNode(root, data) != nullptr;
}

void RedBlackTree::inorder() {
    inorderUtil(root);
}
