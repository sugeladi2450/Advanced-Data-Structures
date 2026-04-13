#ifndef RBTREE_H
#define RBTREE_H

enum Color { RED, BLACK };

struct Node {
    int data;
    Color color;
    Node *parent, *left, *right;

    Node(int data);
};

class RedBlackTree {
private:
    Node* root;

    void rotateLeft(Node* x);
    void rotateRight(Node* x);
    void fixViolation(Node* pt);
    Node* BSTInsert(Node* root, Node* pt);
    void inorderUtil(Node* root);
    Node* searchNode(Node* root, int data) const;
    void clear(Node* root);

public:
    RedBlackTree();
    ~RedBlackTree();
    void insert(const int data);
    bool search(int data) const;
    void inorder();
};

#endif // RBTREE_H
