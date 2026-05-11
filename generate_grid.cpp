#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace std;

int main() {
    const int rows = 1000;   
    const int cols = 1000;

    ofstream f("grid_input.bin", ios::binary);
    f.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    f.write(reinterpret_cast<const char*>(&cols), sizeof(int));

    srand(time(nullptr));
    for (int i = 0; i < rows; i++) {
        vector<double> row(cols);
        for (int j = 0; j < cols; j++)
            row[j] = 20.0 + (rand() % 10);  // random 20-30
        f.write(reinterpret_cast<const char*>(row.data()),
            cols * sizeof(double));
    }

    f.close();
    cout << "Done! Created grid_input.bin ("
        << rows << "x" << cols << ")\n";
    return 0;
}