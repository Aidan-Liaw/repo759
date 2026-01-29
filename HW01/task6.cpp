#include <iostream>

int main(int argc, char *argv[]) {
    int N = std::stoi(argv[1]);
    for (int idx = 0; idx < N; idx++) {
        printf("%d ", idx);
    }
    printf("%d\n", N);

    for (int idx = N; idx > 0; idx--) {
        std:
            std::cout << idx << ' ';
    }
    std::cout << '0' << std::endl;
}