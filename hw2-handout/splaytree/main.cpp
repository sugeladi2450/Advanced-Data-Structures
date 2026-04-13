#include <iostream>
#include "splaytree.h"

int main() {
    SplayTree tree;

    tree.insert(50);
    tree.insert(30);
    tree.insert(20);
    tree.insert(10);
    tree.insert(25);
    tree.insert(27);
    tree.insert(58);
    tree.insert(54);
    tree.insert(48);

    std::cout << "Inorder traversal of the constructed splay tree:\n";
    tree.inorder();
    std::cout << std::endl;

    return 0;
}
