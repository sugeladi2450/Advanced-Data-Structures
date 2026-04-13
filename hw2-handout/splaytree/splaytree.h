#ifndef SPLAYTREE_H
#define SPLAYTREE_H

struct SplayNode {
    int data;
    SplayNode *parent, *left, *right;

    explicit SplayNode(int data);
};

class SplayTree {
private:
    SplayNode* root;

    void rotateLeft(SplayNode* x);
    void rotateRight(SplayNode* x);
    void splay(SplayNode* x);
    void inorderUtil(SplayNode* root) const;
    void clear(SplayNode* root);

public:
    SplayTree();
    ~SplayTree();

    void insert(int data);
    bool search(int data);
    void inorder() const;
};

#endif // SPLAYTREE_H
