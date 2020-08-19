#include "tsne.h"
#include <math.h>
#include <iostream>
#include <random>
#include <fstream>
#include <time.h>
#include <bits/stdc++.h>
#include <pthread.h>

double inline dist(double X1[], double X2[], double sigma)
// return distance under RBF kernel
{
    double sum = 0;
    for (int j = 0; j != PCA_SIZE; j++)
        sum -= pow((X1[j] - X2[j]), 2);
    return exp(sum / 2 / sigma / sigma);
}

double inline t_dist(double Y1[], double Y2[])
// return distance under Cauchy distribution 1/(1+||y1-y2||^2)
{
    double sum = 1;
    for (int j = 0; j != Y_SIZE; j++)
        sum += pow((Y1[j] - Y2[j]), 2);
    return 1 / sum;
}



void find_sigma(double X[][PCA_SIZE], double P[][DATA_SIZE], int i)
{
    int perplexity = 30;
    double varepsilon = 0.01;
    int max_iteration = 50;
    double sigma_min = 0.5;
    double sigma_max = 3;
    double sigma = 1;
    int counts = 0;
    double p_j_cond_i_with_i[DATA_SIZE];
    for (;;)
    {
        double exp_j_cond_i[DATA_SIZE];
        double sum = 0;
        for (int n = 0; n != DATA_SIZE; n++)
        {
            exp_j_cond_i[n] = dist(X[n], X[i], sigma);
            sum += exp_j_cond_i[n];
        }
        sum -= 1;
        double H_P_i = 0;
        for (int n = 0; n != DATA_SIZE; n++)
        {
            p_j_cond_i_with_i[n] = exp_j_cond_i[n] / sum;
            if (n != i)
                H_P_i += -p_j_cond_i_with_i[n] * log(p_j_cond_i_with_i[n]);
        }
        double perp = exp(H_P_i);
        if (perp < perplexity - varepsilon)
        {
            sigma_min = sigma;
            sigma = (sigma_max + sigma_min) / 2;
        }
        else if (perp > perplexity + varepsilon)
        {
            sigma_max = sigma;
            sigma = (sigma_max + sigma_min) / 2;
        }
        else
            break;
        counts += 1;
        if (counts > max_iteration)
            break;
    }
    for (int n = 0; n != DATA_SIZE; n++)
        P[i][n] = p_j_cond_i_with_i[n];
    P[i][i] = 1;
    //std::cout << sigma << std::endl;
}

void * find_sigma_thread_caller(void * arg)
{
    Sigma_arg * ptr = (Sigma_arg * ) arg;
    std::cout << ptr->i << std::endl;
    for (int j=0; j!=ptr->block_size; j++)
    {
        find_sigma(ptr->X, ptr->P, ptr->i + j);
        if (j % 1000==0)
            std::cout << "another 1000 is done\n";
    }
}

void compute_similarity(double X[][PCA_SIZE], double P[][DATA_SIZE], int thread_n)
{
    int block_size = DATA_SIZE / thread_n;
    for (int i = 0; i != DATA_SIZE; i++)
    {
        find_sigma(X, P, i);
        if (i % 1000 == 0)
            std::cout << "similarity compute finished for " << i << " points\n";
    }
    /*
    pthread_t *pt = new pthread_t[thread_n];
    Sigma_arg arg[thread_n];
    for (int i=0; i!=thread_n; i++)
    {
        Sigma_arg arg{X, P, i * block_size, block_size};
        pthread_create(&pt[i], NULL, find_sigma_thread_caller, & arg);
    }
    for (int i = 0; i < thread_n; ++i) pthread_join(pt[i], NULL);
    */
    for (int i = 0; i != DATA_SIZE; i++)
        for (int j = 0; j != i; j++)
        {
            double temp = (P[i][j] + P[j][i]) / 2 / DATA_SIZE;
            P[i][j] = temp;
            P[j][i] = temp;
        }
    //delete [] pt;
};

void init(double Y[][Y_SIZE])
// initialize Y
{
    double multiplier = 0.0001;
    std::default_random_engine gen(233);
    std::normal_distribution<double> dis(0, 1);
    for (int i = 0; i != 1000; i++)
    {
        Y[i][0] = dis(gen) * multiplier;
        Y[i][1] = dis(gen) * multiplier;
    }
}

void compute_Q(double Q[DATA_SIZE][DATA_SIZE], double Y[DATA_SIZE][Y_SIZE])
{
    double sum = 0;
    for (int i = 0; i != DATA_SIZE; i++) // CAUTION:the diagonal term is not set in this loop!
        for (int j = 0; j != i; j++)
        {
            Q[i][j] = t_dist(Y[i], Y[j]);
            Q[j][i] = Q[i][j];
            sum += Q[i][j];
        }
    for (int i = 0; i != DATA_SIZE; i++)
        for (int j = 0; j != i; j++)
        {
            Q[i][j] = Q[i][j] / sum;
            Q[j][i] = Q[j][i] / sum;
        }
}

void update(double P[][DATA_SIZE], double Y[][Y_SIZE], int step_count, std::vector<Log> &log)
// do single step update
{
    static clock_t starttime;
    if (step_count == 0)
        starttime = clock();
    static double Y_old[DATA_SIZE][Y_SIZE] = {0};
    // parameters
    int P_multiplier;
    if (step_count < 100) //early exaggeration
        P_multiplier = 4;
    else
        P_multiplier = 1;
    double alpha = 0.5;
    int eta = 100;
    double Q[DATA_SIZE][DATA_SIZE] = {0};
    double grad_y[DATA_SIZE][Y_SIZE] = {0};
    compute_Q(Q, Y);
    for (int i = 0; i != DATA_SIZE; i++) // compute gradient for y_i
    {
        double sum[2] = {0, 0};
        for (int j = 0; j != DATA_SIZE; j++) // sum over J
        {
            sum[0] += 4 * (P[i][j] * P_multiplier - Q[i][j]) * t_dist(Y[i], Y[j]) * (Y[i][0] - Y[j][0]);
            sum[1] += 4 * (P[i][j] * P_multiplier - Q[i][j]) * t_dist(Y[i], Y[j]) * (Y[i][1] - Y[j][1]);
        }
        grad_y[i][0] = sum[0];
        grad_y[i][1] = sum[1];
    }
    for (int i = 0; i != DATA_SIZE; i++) // GD for y_i
    {
        Y[i][0] = Y[i][0] - eta * grad_y[i][0] + alpha * (Y[i][0] - Y_old[i][0]);
        Y[i][1] = Y[i][1] - eta * grad_y[i][1] + alpha * (Y[i][1] - Y_old[i][1]);
        Y_old[i][0] = Y[i][0];
        Y_old[i][1] = Y[i][1];
    }
    // logging
    if (step_count % 10 == 0)
    {
        double totaltime = double(clock() - starttime) / CLOCKS_PER_SEC;
        Log mylog = {step_count, totaltime, KL(P, Q)};
        log.push_back(mylog);
    }
    std::cout << "step:" << step_count << " completes.\n";
}

double KL(double P[][DATA_SIZE], double Q[][DATA_SIZE])
{
    double KL_sum = 0;
    int count = 0;
    for (int i = 0; i != DATA_SIZE; i++)
        for (int j = 0; j != DATA_SIZE; j++)
        {
            count++;
            if (i != j)
                KL_sum += P[i][j] * log(P[i][j] / Q[i][j]);
            if (count % 43 == 0)
                int t = 1;
        };
    return KL_sum;
}

void write_result_to_file(double Y[][Y_SIZE], std::vector<Log> log)
{
    std::ofstream outfile;
    outfile.open("result_Y.txt");
    for (int i = 0; i != DATA_SIZE; i++)
        outfile << Y[i][0] << " " << Y[i][1] << std::endl;
    outfile.close();
    std::ofstream outfile2;
    outfile2.open("result_log.txt");
    for (auto i : log)
        outfile2 << i.iteration << " " << i.time << " " << i.KL_div << std::endl;
    outfile2.close();
}

void BCD_update(double P[][DATA_SIZE], double Y[][Y_SIZE], int step_count, std::vector<Log> &log, int thread_n, int block_n)
// do single step update for block coordinate descent
{
    static clock_t starttime;
    int block_size = DATA_SIZE / block_n;
    if (step_count == 0)
        starttime = clock();
    double(*Q)[DATA_SIZE] = new double[DATA_SIZE][DATA_SIZE];
    //1. split random
    int *perm = new int[DATA_SIZE];
    //int * perm = new int[10];
    for (int i = 0; i != DATA_SIZE; i++)
        perm[i] = i;
    std::random_shuffle(perm, perm + DATA_SIZE);
    static double Y_old[DATA_SIZE][Y_SIZE] = {0};
    //2. open threading
    /*for (int t = 0; t != block_n; t++)
    {
        BCD_each(perm + t * block_size, P, Q, Y, Y_old, block_n, step_count);
    }
    */
    pthread_t *pt = new pthread_t[thread_n];
    common_BCD_thread_arg common_arg{P, Q, Y, Y_old, log, block_n};
    for (int i = 0; i != thread_n; i++)
    {
        BCD_thread_arg bcd_arg{perm + i * block_size, step_count, common_arg};
        pthread_create(&pt[i], NULL, BCD_thread_caller, & bcd_arg);
    }
    for (int i = 0; i < thread_n; ++i) pthread_join(pt[i], NULL);
    //3. wait for stop
    if (step_count % 20 == 0)
    {
        double totaltime = double(clock() - starttime) / CLOCKS_PER_SEC;
        compute_Q(Q, Y);
        Log mylog = {step_count, totaltime, KL(P, Q)};
        log.push_back(mylog);
        std::cout << "step:" << step_count << "\tloss" << log.back().KL_div << std::endl ;
    }
    //std::cout << "step:" << step_count << " completes.\n";
    delete[] perm;
    delete[] Q;
    delete[] pt;
}

void BCD_each(int *index, double P[][DATA_SIZE], double Q[][DATA_SIZE], double Y[][Y_SIZE], double Y_old[][Y_SIZE], int block_n, int step_count)
{
    int block_size = DATA_SIZE / block_n;
    compute_block_Q(index, Q, Y, block_size);
    double(*grad_y)[Y_SIZE] = new double[DATA_SIZE][Y_SIZE];
    int P_multiplier;
    if (step_count < 100) //early exaggeration
        P_multiplier = 4;
    else
        P_multiplier = 1;
    double alpha = 0.5;
    int eta = 100;
    for (int i = 0; i != block_size; i++) // compute gradient for y_i
    {
        int ii = index[i];
        double sum[2] = {0, 0};
        for (int j = 0; j != block_size; j++) // sum over J
        {
            int jj = index[j];
            sum[0] += 4 * (P[ii][jj] * P_multiplier * block_n * block_n - Q[ii][jj]) * t_dist(Y[ii], Y[jj]) * (Y[ii][0] - Y[jj][0]);
            sum[1] += 4 * (P[ii][jj] * P_multiplier * block_n * block_n - Q[ii][jj]) * t_dist(Y[ii], Y[jj]) * (Y[ii][1] - Y[jj][1]);
        }
        grad_y[ii][0] = sum[0];
        grad_y[ii][1] = sum[1];
    }
    for (int i = 0; i != block_size; i++) // GD for y_i
    {
        int ii = index[i];
        Y[ii][0] = Y[ii][0] - eta * grad_y[ii][0] + alpha * (Y[ii][0] - Y_old[ii][0]);
        Y[ii][1] = Y[ii][1] - eta * grad_y[ii][1] + alpha * (Y[ii][1] - Y_old[ii][1]);
        Y_old[ii][0] = Y[ii][0];
        Y_old[ii][1] = Y[ii][1];
    }
    delete[] grad_y;
}

void compute_block_Q(int *index, double Q[DATA_SIZE][DATA_SIZE], double Y[DATA_SIZE][Y_SIZE], int block_size)
{
    double sum = 0;
    for (int i = 0; i != block_size; i++) // CAUTION:the diagonal term is not set in this loop!
        for (int j = 0; j != i; j++)
        {
            Q[index[i]][index[j]] = t_dist(Y[index[i]], Y[index[j]]);
            Q[index[j]][index[i]] = Q[index[i]][index[j]];
            sum += Q[index[i]][index[j]];
        }
    for (int i = 0; i != block_size; i++)
        for (int j = 0; j != i; j++)
        {
            Q[index[i]][index[j]] = Q[index[i]][index[j]] / sum;
            Q[index[j]][index[i]] = Q[index[j]][index[i]] / sum;
        }
}

void *BCD_thread_caller(void *arg)
{
    BCD_thread_arg *argptr = (BCD_thread_arg *)arg;
    BCD_each(argptr->index, argptr->common_arg.P, argptr->common_arg.Q, argptr->common_arg.Y, argptr->common_arg.Y_old, argptr->common_arg.block_n, argptr->step_count);
}

common_BCD_thread_arg::common_BCD_thread_arg(double p[][DATA_SIZE], double q[][DATA_SIZE], double y[][Y_SIZE], double y_old[][Y_SIZE], std::vector<Log> &l, int b) : P(p), Q(q), Y(y), Y_old(y_old), log(l), block_n(b) {}
BCD_thread_arg::BCD_thread_arg(int *i, int s, common_BCD_thread_arg carg) : index(i), step_count(s), common_arg(carg) {}
Sigma_arg::Sigma_arg(double x[][PCA_SIZE],double p[][DATA_SIZE], int I, int bs) : X(x), P(p), i(I), block_size(bs) {} ;