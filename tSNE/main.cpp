#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
#include <vector>
#include "tsne.h"

using namespace std;

void read_file(double X[][PCA_SIZE])
{
    ifstream infile;
    infile.open("X_pca_10000.txt", ios::in);
    for (int i = 0; i != DATA_SIZE; i++)
    {
        stringstream ss;
        string line;
        getline(infile, line);
        ss << line;
        for (int j = 0; j != PCA_SIZE; j++)
        {
            double number;
            ss >> number;
            X[i][j] = number;
        }
        ss.clear();
        //cout << i << " ";
    }
    infile.close();
}

int main(int argc, char **argv)
{
    double (*X_pca)[PCA_SIZE] = new double[DATA_SIZE][PCA_SIZE];
    double (*P)[DATA_SIZE] = new double[DATA_SIZE][DATA_SIZE];
    double (*Y)[Y_SIZE] = new double [DATA_SIZE][Y_SIZE];
    vector<Log> log;
    // initialization
    cout << "reading values\n";
    read_file(X_pca);
    cout << "computing P matrix\n";
    compute_similarity(X_pca, P, 8); // precomputing of P_ij
    init(Y);
    //computing 
    cout << "start training\n";
    int n_steps = 1000;
    int step_count = 0;
    int block_n = 5;
    int thread_n = block_n;
    for (;step_count!=n_steps; step_count++ )
        //update(P, Y, step_count, log);
        BCD_update(P, Y, step_count,log, thread_n, block_n);
    write_result_to_file(Y, log);
    delete [] X_pca;
    delete [] Y;
    delete [] P;
    return 0;
}
