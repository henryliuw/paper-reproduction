#ifndef __TSNE_H
#define __TSNE_H

#ifndef PCA_SIZE
#define PCA_SIZE 50
#endif
#ifndef DATA_SIZE
#define DATA_SIZE 10000
#endif
#ifndef Y_SIZE
#define Y_SIZE 2
#endif
#include <vector>

class Log
{
public:
    int iteration;
    double time;
    double KL_div;
};
class Sigma_arg
{
public:
    double (*X)[PCA_SIZE];
    double (*P)[DATA_SIZE]; 
    int i;
    int block_size;
    Sigma_arg();
    Sigma_arg(double x[][PCA_SIZE],double p[][DATA_SIZE], int I, int bs);
};
class common_BCD_thread_arg
{
public:
    double (*P) [DATA_SIZE];
    double (*Q) [DATA_SIZE];
    double (*Y) [Y_SIZE];
    double (*Y_old)[Y_SIZE];
    std::vector<Log> &log;
    int block_n;
    common_BCD_thread_arg(double p[][DATA_SIZE], double q[][DATA_SIZE], double y[][Y_SIZE],  double y_old[][Y_SIZE],  std::vector<Log> &l, int b);
};
class BCD_thread_arg
{
public:
    int *index;
    int step_count;
    common_BCD_thread_arg common_arg;
    BCD_thread_arg(int *i, int s, common_BCD_thread_arg carg);
};
void compute_similarity(double X[][PCA_SIZE], double P[][DATA_SIZE], int thread_n);
void find_sigma(double X[][PCA_SIZE], double P[][DATA_SIZE], int i);
double dist(double X1[], double X2[], double sigma);
void init(double Y[][Y_SIZE]);
void compute_Q(double Q[][DATA_SIZE], double Y[][Y_SIZE]);
void update(double P[][DATA_SIZE], double Y[][Y_SIZE], int step_count, std::vector<Log> &log);
void write_result_to_file(double Y[][Y_SIZE], std::vector<Log> log);
double KL(double P[][DATA_SIZE], double Q[][DATA_SIZE]);
void BCD_update(double P[][DATA_SIZE], double Y[][Y_SIZE], int step_count, std::vector<Log> &log, int thread_n, int block_n);
void BCD_each(int *index, double P[][DATA_SIZE], double Q[][DATA_SIZE], double Y[][Y_SIZE], double Y_old[][Y_SIZE], int block_n, int step_count);
void compute_block_Q(int *index, double Q[DATA_SIZE][DATA_SIZE], double Y[DATA_SIZE][Y_SIZE], int block_size);
void * BCD_thread_caller(void * arg);

#endif