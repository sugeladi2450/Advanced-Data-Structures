#include <iostream>
#include "rbtree.h"
int main() {
    RedBlackTree tree;

    tree.insert(50);
    tree.insert(30);
    tree.insert(20);
    tree.insert(10);
    tree.insert(25);
    tree.insert(27);
    tree.insert(58);
    tree.insert(54);
    tree.insert(48);

    std::cout << "Inorder traversal of the constructed tree (value(color)): \n";
    tree.inorder();
    std::cout << std::endl;

    return 0;
}
