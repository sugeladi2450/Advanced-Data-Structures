#include "splaytree.h"
#include <iostream>

SplayNode::SplayNode(int data)
    : data(data), parent(nullptr), left(nullptr), right(nullptr) {}

SplayTree::SplayTree() : root(nullptr) {}

SplayTree::~SplayTree() {
    clear(root);
}

void SplayTree::rotateLeft(SplayNode* x) {
    SplayNode* y = x->right;

    if (y == nullptr)
        return;

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

void SplayTree::rotateRight(SplayNode* x) {
    SplayNode* y = x->left;

    if (y == nullptr)
        return;

    x->left = y->right;
    if (y->right != nullptr)
        y->right->parent = x;

    y->parent = x->parent;
    if (x->parent == nullptr)
        root = y;
    else if (x == x->parent->left)
        x->parent->left = y;
    else
        x->parent->right = y;

    y->right = x;
    x->parent = y;
}

void SplayTree::splay(SplayNode* x) {
    while (x != nullptr && x->parent != nullptr) {
        SplayNode* parent = x->parent;
        SplayNode* grand_parent = parent->parent;

        if (grand_parent == nullptr) {
            if (x == parent->left)
                rotateRight(parent);
            else
                rotateLeft(parent);
        } else if (x == parent->left && parent == grand_parent->left) {
            rotateRight(grand_parent);
            rotateRight(parent);
        } else if (x == parent->right && parent == grand_parent->right) {
            rotateLeft(grand_parent);
            rotateLeft(parent);
        } else if (x == parent->right && parent == grand_parent->left) {
            rotateLeft(parent);
            rotateRight(grand_parent);
        } else {
            rotateRight(parent);
            rotateLeft(grand_parent);
        }
    }
}

void SplayTree::inorderUtil(SplayNode* root) const {
    if (root == nullptr)
        return;

    inorderUtil(root->left);
    std::cout << root->data << " ";
    inorderUtil(root->right);
}

void SplayTree::clear(SplayNode* root) {
    if (root == nullptr)
        return;

    clear(root->left);
    clear(root->right);
    delete root;
}

void SplayTree::insert(int data) {
    if (root == nullptr) {
        root = new SplayNode(data);
        return;
    }

    SplayNode* current = root;
    SplayNode* parent = nullptr;

    while (current != nullptr) {
        parent = current;

        if (data < current->data)
            current = current->left;
        else if (data > current->data)
            current = current->right;
        else {
            splay(current);
            return;
        }
    }

    SplayNode* new_node = new SplayNode(data);
    new_node->parent = parent;

    if (data < parent->data)
        parent->left = new_node;
    else
        parent->right = new_node;

    splay(new_node);
}

bool SplayTree::search(int data) {
    SplayNode* current = root;
    SplayNode* last = nullptr;

    while (current != nullptr) {
        last = current;

        if (data < current->data)
            current = current->left;
        else if (data > current->data)
            current = current->right;
        else {
            splay(current);
            return true;
        }
    }

    if (last != nullptr)
        splay(last);

    return false;
}

void SplayTree::inorder() const {
    inorderUtil(root);
}
