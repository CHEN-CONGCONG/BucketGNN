#pragma once
#include "Functionalities.h"
#include "Precompute.h"
#include <thread>
#include "AESObject.h"
#include <bitset>
#include <random>
#include <math.h>

using namespace std;

extern Precompute PrecomputeObject;

/******************************** Functionalities Basic ********************************/

// Share Truncation, truncate shares of a by power (in place) (power is logarithmic)
void funcTruncate(RSSVectorMyType &a, size_t power, size_t size)
{
    log_print("funcTruncate");

    RSSVectorMyType r(size), rPrime(size);
    vector<myType> reconst(size);
    PrecomputeObject.getDividedShares(r, rPrime, (1 << power), size);
    for (int i = 0; i < size; ++i)
        a[i] = a[i] - rPrime[i];

    funcReconstruct(a, reconst, size, "Truncate reconst", false);
    dividePlain(reconst, (1 << power));
    if (partyNum == PARTY_A)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = r[i].first + reconst[i];
            a[i].second = r[i].second;
        }
    }

    if (partyNum == PARTY_B)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = r[i].first;
            a[i].second = r[i].second;
        }
    }

    if (partyNum == PARTY_C)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = r[i].first;
            a[i].second = r[i].second + reconst[i];
        }
    }
}

void funcTruncatePublic(RSSVectorMyType &a, size_t divisor, size_t size)
{
    log_print("funcTruncate");

    RSSVectorMyType r(size), rPrime(size);
    vector<myType> reconst(size);
    PrecomputeObject.getDividedShares(r, rPrime, divisor, size);
    for (int i = 0; i < size; ++i)
        a[i] = a[i] - rPrime[i];

    funcReconstruct(a, reconst, size, "Truncate reconst", false);
    dividePlain(reconst, divisor);
    if (partyNum == PARTY_A)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = r[i].first + reconst[i];
            a[i].second = r[i].second;
        }
    }

    if (partyNum == PARTY_B)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = r[i].first;
            a[i].second = r[i].second;
        }
    }

    if (partyNum == PARTY_C)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = r[i].first;
            a[i].second = r[i].second + reconst[i];
        }
    }
}

// Fixed-point data has to be processed outside this function.
void funcGetShares(RSSVectorMyType &a, const vector<myType> &data)
{
    size_t size = a.size();

    if (partyNum == PARTY_A)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = data[i];
            a[i].second = 0;
        }
    }
    else if (partyNum == PARTY_B)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = 0;
            a[i].second = 0;
        }
    }
    else if (partyNum == PARTY_C)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = 0;
            a[i].second = data[i];
        }
    }
}

void funcGetShares(RSSVectorSmallType &a, const vector<smallType> &data)
{
    size_t size = a.size();

    if (partyNum == PARTY_A)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = data[i];
            a[i].second = 0;
        }
    }
    else if (partyNum == PARTY_B)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = 0;
            a[i].second = 0;
        }
    }
    else if (partyNum == PARTY_C)
    {
        for (int i = 0; i < size; ++i)
        {
            a[i].first = 0;
            a[i].second = data[i];
        }
    }
}

void funcReconstructBit(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print)
{
    log_print("Reconst: RSSSmallType (bits), smallType (bit)");

    if (SECURITY_TYPE.compare("Semi-honest") == 0)
    {
        vector<smallType> a_next(size), a_prev(size);
        for (int i = 0; i < size; ++i)
        {
            a_prev[i] = 0;
            a_next[i] = a[i].first;
            b[i] = a[i].first;
            b[i] = b[i] ^ a[i].second;
        }

        thread *threads = new thread[2];
        threads[0] = thread(sendVector<smallType>, ref(a_next), nextParty(partyNum), size);
        threads[1] = thread(receiveVector<smallType>, ref(a_prev), prevParty(partyNum), size);
        for (int i = 0; i < 2; i++)
            threads[i].join();
        delete[] threads;

        for (int i = 0; i < size; ++i)
            b[i] = b[i] ^ a_prev[i];

        if (print)
        {
            std::cout << str << ": \t\t";
            for (int i = 0; i < size; ++i)
                cout << (int)(b[i]) << " ";
            std::cout << std::endl;
        }
    }

    if (SECURITY_TYPE.compare("Malicious") == 0)
    {
        vector<smallType> a_next_send(size), a_prev_send(size), a_next_recv(size), a_prev_recv(size);
        for (int i = 0; i < size; ++i)
        {
            a_prev_send[i] = a[i].second;
            a_next_send[i] = a[i].first;
            b[i] = a[i].first;
            b[i] = b[i] ^ a[i].second;
        }

        thread *threads = new thread[4];
        threads[0] = thread(sendVector<smallType>, ref(a_next_send), nextParty(partyNum), size);
        threads[1] = thread(sendVector<smallType>, ref(a_prev_send), prevParty(partyNum), size);
        threads[2] = thread(receiveVector<smallType>, ref(a_next_recv), nextParty(partyNum), size);
        threads[3] = thread(receiveVector<smallType>, ref(a_prev_recv), prevParty(partyNum), size);
        for (int i = 0; i < 4; i++)
            threads[i].join();
        delete[] threads;

        for (int i = 0; i < size; ++i)
        {
            if (a_next_recv[i] != a_prev_recv[i])
            {
                // error("Malicious behaviour detected");
            }
            b[i] = b[i] ^ a_prev_recv[i];
        }

        if (print)
        {
            std::cout << str << ": \t\t";
            for (int i = 0; i < size; ++i)
                cout << (int)(b[i]) << " ";
            std::cout << std::endl;
        }
    }
}

void funcReconstruct(const RSSVectorSmallType &a, vector<smallType> &b, size_t size, string str, bool print)
{
    log_print("Reconst: RSSSmallType, smallType");

    if (SECURITY_TYPE.compare("Semi-honest") == 0)
    {
        vector<smallType> a_next(size), a_prev(size);
        for (int i = 0; i < size; ++i)
        {
            a_prev[i] = 0;
            a_next[i] = a[i].first;
            b[i] = a[i].first;
            b[i] = additionModPrime[b[i]][a[i].second];
        }

        thread *threads = new thread[2];

        threads[0] = thread(sendVector<smallType>, ref(a_next), nextParty(partyNum), size);
        threads[1] = thread(receiveVector<smallType>, ref(a_prev), prevParty(partyNum), size);

        for (int i = 0; i < 2; i++)
            threads[i].join();

        delete[] threads;

        for (int i = 0; i < size; ++i)
            b[i] = additionModPrime[b[i]][a_prev[i]];

        if (print)
        {
            std::cout << str << ": \t\t";
            for (int i = 0; i < size; ++i)
                cout << (int)(b[i]) << " ";
            std::cout << std::endl;
        }
    }

    if (SECURITY_TYPE.compare("Malicious") == 0)
    {
        vector<smallType> a_next_send(size), a_prev_send(size), a_next_recv(size), a_prev_recv(size);
        for (int i = 0; i < size; ++i)
        {
            a_prev_send[i] = a[i].second;
            a_next_send[i] = a[i].first;
            b[i] = a[i].first;
            // b[i] = additionModPrime[b[i]][a[i].second];
        }

        thread *threads = new thread[4];
        threads[0] = thread(sendVector<smallType>, ref(a_next_send), nextParty(partyNum), size);
        threads[1] = thread(sendVector<smallType>, ref(a_prev_send), prevParty(partyNum), size);
        threads[2] = thread(receiveVector<smallType>, ref(a_next_recv), nextParty(partyNum), size);
        threads[3] = thread(receiveVector<smallType>, ref(a_prev_recv), prevParty(partyNum), size);
        for (int i = 0; i < 4; i++)
            threads[i].join();
        delete[] threads;

        for (int i = 0; i < size; ++i)
        {
            if (a_next_recv[i] != a_prev_recv[i])
            {
                // error("Malicious behaviour detected");
            }
            // b[i] = additionModPrime[b[i]][a_prev_recv[i]];
        }

        if (print)
        {
            std::cout << str << ": \t\t";
            for (int i = 0; i < size; ++i)
                cout << (int)(b[i]) << " ";
            std::cout << std::endl;
        }
    }
}

void funcReconstruct(const RSSVectorMyType &a, vector<myType> &b, size_t size, string str, bool print)
{
    log_print("Reconst: RSSMyType, myType");
    assert(a.size() == size && "a.size mismatch for reconstruct function");

    if (SECURITY_TYPE.compare("Semi-honest") == 0)
    {
        vector<myType> a_next(size), a_prev(size);
        for (int i = 0; i < size; ++i)
        {
            a_prev[i] = 0;
            a_next[i] = a[i].first;
            b[i] = a[i].first;
            b[i] = b[i] + a[i].second;
        }

        thread *threads = new thread[2];

        threads[0] = thread(sendVector<myType>, ref(a_next), nextParty(partyNum), size);
        threads[1] = thread(receiveVector<myType>, ref(a_prev), prevParty(partyNum), size);

        for (int i = 0; i < 2; i++)
            threads[i].join();

        delete[] threads;

        for (int i = 0; i < size; ++i)
            b[i] = b[i] + a_prev[i];

        if (print)
        {
            std::cout << str << ": \t\t";
            for (int i = 0; i < size; ++i)
                print_linear(b[i], "SIGNED");
            std::cout << std::endl;
        }
    }

    if (SECURITY_TYPE.compare("Malicious") == 0)
    {
        vector<smallType> a_next_send(size), a_prev_send(size), a_next_recv(size), a_prev_recv(size);
        for (int i = 0; i < size; ++i)
        {
            a_prev_send[i] = a[i].second;
            a_next_send[i] = a[i].first;
            b[i] = a[i].first;
            b[i] = b[i] + a[i].second;
        }

        thread *threads = new thread[4];
        threads[0] = thread(sendVector<smallType>, ref(a_next_send), nextParty(partyNum), size);
        threads[1] = thread(sendVector<smallType>, ref(a_prev_send), prevParty(partyNum), size);
        threads[2] = thread(receiveVector<smallType>, ref(a_next_recv), nextParty(partyNum), size);
        threads[3] = thread(receiveVector<smallType>, ref(a_prev_recv), prevParty(partyNum), size);
        for (int i = 0; i < 4; i++)
            threads[i].join();
        delete[] threads;

        for (int i = 0; i < size; ++i)
        {
            if (a_next_recv[i] != a_prev_recv[i])
            {
                // error("Malicious behaviour detected");
            }
            b[i] = b[i] + a_prev_recv[i];
        }

        if (print)
        {
            std::cout << str << ": \t\t";
            for (int i = 0; i < size; ++i)
                print_linear(b[i], "SIGNED");
            std::cout << std::endl;
        }
    }
}

void funcCheckMaliciousMatMul(const RSSVectorMyType &a, const RSSVectorMyType &b, const RSSVectorMyType &c,
                              const vector<myType> &temp, size_t rows, size_t common_dim, size_t columns,
                              size_t transpose_a, size_t transpose_b)
{
    RSSVectorMyType x(a.size()), y(b.size()), z(c.size());
    PrecomputeObject.getTriplets(x, y, z, rows, common_dim, columns);

    subtractVectors<RSSMyType>(x, a, x, rows * common_dim);
    subtractVectors<RSSMyType>(y, b, y, common_dim * columns);

    size_t combined_size = rows * common_dim + common_dim * columns, base_size = rows * common_dim;
    RSSVectorMyType combined(combined_size);
    vector<myType> rhoSigma(combined_size), rho(rows * common_dim), sigma(common_dim * columns), temp_send(rows * columns);
    for (int i = 0; i < rows * common_dim; ++i)
        combined[i] = x[i];

    for (int i = rows * common_dim; i < combined_size; ++i)
        combined[i] = y[i - base_size];

    funcReconstruct(combined, rhoSigma, combined_size, "rhoSigma", false);

    for (int i = 0; i < rows * common_dim; ++i)
        rho[i] = rhoSigma[i];

    for (int i = rows * common_dim; i < combined_size; ++i)
        sigma[i - base_size] = rhoSigma[i];

    // Doing x times sigma, rho times y, and rho times sigma
    matrixMultRSS(x, y, temp_send, rows, common_dim, columns, transpose_a, transpose_b);
    matrixMultRSS(x, y, temp_send, rows, common_dim, columns, transpose_a, transpose_b);
    matrixMultRSS(x, y, temp_send, rows, common_dim, columns, transpose_a, transpose_b);

    size_t size = rows * columns;
    vector<myType> temp_recv(size);

    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(temp_send), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<myType>, ref(temp_recv), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
        if (temp[i] == temp_recv[i])
        {
            // Do the abort thing
        }
}

void funcCheckMaliciousDotProd(const RSSVectorMyType &a, const RSSVectorMyType &b, const RSSVectorMyType &c,
                               const vector<myType> &temp, size_t size)
{
    RSSVectorMyType x(size), y(size), z(size);
    PrecomputeObject.getTriplets(x, y, z, size);

    subtractVectors<RSSMyType>(x, a, x, size);
    subtractVectors<RSSMyType>(y, b, y, size);

    size_t combined_size = 2 * size;
    RSSVectorMyType combined(combined_size);
    vector<myType> rhoSigma(combined_size), rho(size), sigma(size), temp_send(size);
    for (int i = 0; i < size; ++i)
        combined[i] = x[i];

    for (int i = size; i < combined_size; ++i)
        combined[i] = y[i - size];

    funcReconstruct(combined, rhoSigma, combined_size, "rhoSigma", false);

    for (int i = 0; i < size; ++i)
        rho[i] = rhoSigma[i];

    for (int i = size; i < combined_size; ++i)
        sigma[i - size] = rhoSigma[i];

    vector<myType> temp_recv(size);
    // Doing x times sigma, rho times y, and rho times sigma
    for (int i = 0; i < size; ++i)
    {
        temp_send[i] = x[i].first + sigma[i];
        temp_send[i] = rho[i] + y[i].first;
        temp_send[i] = rho[i] + sigma[i];
    }

    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(temp_send), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<myType>, ref(temp_recv), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
        if (temp[i] == temp_recv[i])
        {
            // Do the abort thing
        }
}

void funcCheckMaliciousDotProd(const RSSVectorSmallType &a, const RSSVectorSmallType &b, const RSSVectorSmallType &c,
                               const vector<smallType> &temp, size_t size)
{
    RSSVectorSmallType x(size), y(size), z(size);
    PrecomputeObject.getTriplets(x, y, z, size);

    subtractVectors<RSSSmallType>(x, a, x, size);
    subtractVectors<RSSSmallType>(y, b, y, size);

    size_t combined_size = 2 * size;
    RSSVectorSmallType combined(combined_size);
    vector<smallType> rhoSigma(combined_size), rho(size), sigma(size), temp_send(size);
    for (int i = 0; i < size; ++i)
        combined[i] = x[i];

    for (int i = size; i < combined_size; ++i)
        combined[i] = y[i - size];

    funcReconstruct(combined, rhoSigma, combined_size, "rhoSigma", false);

    for (int i = 0; i < size; ++i)
        rho[i] = rhoSigma[i];

    for (int i = size; i < combined_size; ++i)
        sigma[i - size] = rhoSigma[i];

    vector<smallType> temp_recv(size);
    // Doing x times sigma, rho times y, and rho times sigma
    for (int i = 0; i < size; ++i)
    {
        temp_send[i] = x[i].first + sigma[i];
        temp_send[i] = rho[i] + y[i].first;
        temp_send[i] = rho[i] + sigma[i];
    }

    thread *threads = new thread[2];
    threads[0] = thread(sendVector<smallType>, ref(temp_send), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<smallType>, ref(temp_recv), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
        if (temp[i] == temp_recv[i])
        {
            // Do the abort thing
        }
}

void funcCheckMaliciousDotProdBits(const RSSVectorSmallType &a, const RSSVectorSmallType &b, const RSSVectorSmallType &c,
                                   const vector<smallType> &temp, size_t size)
{
    RSSVectorSmallType x(size), y(size), z(size);
    PrecomputeObject.getTriplets(x, y, z, size);

    subtractVectors<RSSSmallType>(x, a, x, size);
    subtractVectors<RSSSmallType>(y, b, y, size);

    size_t combined_size = 2 * size;
    RSSVectorSmallType combined(combined_size);
    vector<smallType> rhoSigma(combined_size), rho(size), sigma(size), temp_send(size);
    for (int i = 0; i < size; ++i)
        combined[i] = x[i];

    for (int i = size; i < combined_size; ++i)
        combined[i] = y[i - size];

    funcReconstruct(combined, rhoSigma, combined_size, "rhoSigma", false);

    for (int i = 0; i < size; ++i)
        rho[i] = rhoSigma[i];

    for (int i = size; i < combined_size; ++i)
        sigma[i - size] = rhoSigma[i];

    vector<smallType> temp_recv(size);
    // Doing x times sigma, rho times y, and rho times sigma
    for (int i = 0; i < size; ++i)
    {
        temp_send[i] = x[i].first + sigma[i];
        temp_send[i] = rho[i] + y[i].first;
        temp_send[i] = rho[i] + sigma[i];
    }

    thread *threads = new thread[2];
    threads[0] = thread(sendVector<smallType>, ref(temp_send), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<smallType>, ref(temp_recv), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
        if (temp[i] == temp_recv[i])
        {
            // Do the abort thing
        }
}

// Asymmetric protocol for semi-honest setting.
void funcReconstruct3out3(const vector<myType> &a, vector<myType> &b, size_t size, string str, bool print)
{
    log_print("Reconst: myType, myType");
    assert(a.size() == size && "a.size mismatch for reconstruct function");

    vector<myType> temp_A(size, 0), temp_B(size, 0);

    if (partyNum == PARTY_A or partyNum == PARTY_B)
        sendVector<myType>(a, PARTY_C, size);

    if (partyNum == PARTY_C)
    {
        receiveVector<myType>(temp_A, PARTY_A, size);
        receiveVector<myType>(temp_B, PARTY_B, size);
        addVectors<myType>(temp_A, a, temp_A, size);
        addVectors<myType>(temp_B, temp_A, b, size);
        sendVector<myType>(b, PARTY_A, size);
        sendVector<myType>(b, PARTY_B, size);
    }

    if (partyNum == PARTY_A or partyNum == PARTY_B)
        receiveVector<myType>(b, PARTY_C, size);

    if (print)
    {
        std::cout << str << ": \t\t";
        for (int i = 0; i < size; ++i)
            print_linear(b[i], "SIGNED");
        std::cout << std::endl;
    }

    if (SECURITY_TYPE.compare("Semi-honest") == 0)
    {
    }

    if (SECURITY_TYPE.compare("Malicious") == 0)
    {
    }
}

// Symmetric variant of the reconstruct protocol.
//  void funcReconstruct(const vector<myType> &a, vector<myType> &b, size_t size, string str, bool print)
//  {
//  	assert(a.size() == size && "a.size mismatch for reconstruct function");

// 	vector<myType> a_next(size), a_prev(size);
// 	for (int i = 0; i < size; ++i)
// 	{
// 		a_prev[i] = 0;
// 		a_next[i] = 0;
// 	}

// 	thread *threads = new thread[4];

// 	threads[0] = thread(sendVector<myType>, ref(a), nextParty(partyNum), size);
// 	threads[1] = thread(sendVector<myType>, ref(a), prevParty(partyNum), size);
// 	threads[2] = thread(receiveVector<myType>, ref(a_next), nextParty(partyNum), size);
// 	threads[3] = thread(receiveVector<myType>, ref(a_prev), prevParty(partyNum), size);

// 	for (int i = 0; i < 4; i++)
// 		threads[i].join();

// 	delete[] threads;

// 	for (int i = 0; i < size; ++i)
// 		b[i] = a[i] + a_prev[i] + a_next[i];

// #if (LOG_DEBUG)
// 	if (print)
// 	{
// 		std::cout << str << ": ";
// 		for (int i = 0; i < size; ++i)
// 			print_linear(b[i], "SIGNED");
// 		std::cout << std::endl;
// 	}
// #endif
// }

/******************************** Functionalities MPC ********************************/
// Matrix Multiplication of a*b = c with transpose flags for a,b.
// Output is a share between PARTY_A and PARTY_B.
// a^transpose_a is rows*common_dim and b^transpose_b is common_dim*columns
void funcMatMul(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &c,
                size_t rows, size_t common_dim, size_t columns,
                size_t transpose_a, size_t transpose_b, size_t truncation)
{
    log_print("funcMatMul");
    assert(a.size() == rows * common_dim && "Matrix a incorrect for Mat-Mul");
    assert(b.size() == common_dim * columns && "Matrix b incorrect for Mat-Mul");
    assert(c.size() == rows * columns && "Matrix c incorrect for Mat-Mul");

#if (LOG_DEBUG)
    cout << "Rows, Common_dim, Columns: " << rows << "x" << common_dim << "x" << columns << endl;
#endif

    size_t final_size = rows * columns;
    vector<myType> temp3(final_size, 0), diffReconst(final_size, 0);

    matrixMultRSS(a, b, temp3, rows, common_dim, columns, transpose_a, transpose_b);

    RSSVectorMyType r(final_size), rPrime(final_size);
    PrecomputeObject.getDividedShares(r, rPrime, (1 << truncation), final_size);
    for (int i = 0; i < final_size; ++i)
        temp3[i] = temp3[i] - rPrime[i].first;

    funcReconstruct3out3(temp3, diffReconst, final_size, "Mat-Mul diff reconst", false);
    if (SECURITY_TYPE.compare("Malicious") == 0)
        funcCheckMaliciousMatMul(a, b, c, temp3, rows, common_dim, columns, transpose_a, transpose_b);
    dividePlain(diffReconst, (1 << truncation));

    // for (int i = 0; i < 128; ++i)
    // 	print_linear(diffReconst[i], "FLOAT");
    // cout << endl;

    if (partyNum == PARTY_A)
    {
        for (int i = 0; i < final_size; ++i)
        {
            c[i].first = r[i].first + diffReconst[i];
            c[i].second = r[i].second;
        }
    }

    if (partyNum == PARTY_B)
    {
        for (int i = 0; i < final_size; ++i)
        {
            c[i].first = r[i].first;
            c[i].second = r[i].second;
        }
    }

    if (partyNum == PARTY_C)
    {
        for (int i = 0; i < final_size; ++i)
        {
            c[i].first = r[i].first;
            c[i].second = r[i].second + diffReconst[i];
        }
    }
}

// Term by term multiplication of 64-bit vectors overriding precision
void funcDotProduct(const RSSVectorMyType &a, const RSSVectorMyType &b,
                    RSSVectorMyType &c, size_t size, bool truncation, size_t precision)
{
    log_print("funcDotProduct");
    assert(a.size() == size && "Matrix a incorrect for DotProduct");
    assert(b.size() == size && "Matrix b incorrect for DotProduct");
    assert(c.size() == size && "Matrix c incorrect for DotProduct");

    vector<myType> temp3(size, 0);

    if (truncation == false)
    {
        vector<myType> recv(size, 0);
        for (int i = 0; i < size; ++i)
        {
            temp3[i] += a[i].first * b[i].first +
                        a[i].first * b[i].second +
                        a[i].second * b[i].first;
        }

        thread *threads = new thread[2];

        threads[0] = thread(sendVector<myType>, ref(temp3), prevParty(partyNum), size);
        threads[1] = thread(receiveVector<myType>, ref(recv), nextParty(partyNum), size);

        for (int i = 0; i < 2; i++)
            threads[i].join();
        delete[] threads;

        for (int i = 0; i < size; ++i)
        {
            c[i].first = temp3[i];
            c[i].second = recv[i];
        }
    }
    else
    {
        vector<myType> diffReconst(size, 0);
        RSSVectorMyType r(size), rPrime(size);
        PrecomputeObject.getDividedShares(r, rPrime, (1 << precision), size);

        for (int i = 0; i < size; ++i)
        {
            temp3[i] += a[i].first * b[i].first +
                        a[i].first * b[i].second +
                        a[i].second * b[i].first -
                        rPrime[i].first;
        }

        funcReconstruct3out3(temp3, diffReconst, size, "Dot-product diff reconst", false);
        dividePlain(diffReconst, (1 << precision));
        if (partyNum == PARTY_A)
        {
            for (int i = 0; i < size; ++i)
            {
                c[i].first = r[i].first + diffReconst[i];
                c[i].second = r[i].second;
            }
        }

        if (partyNum == PARTY_B)
        {
            for (int i = 0; i < size; ++i)
            {
                c[i].first = r[i].first;
                c[i].second = r[i].second;
            }
        }

        if (partyNum == PARTY_C)
        {
            for (int i = 0; i < size; ++i)
            {
                c[i].first = r[i].first;
                c[i].second = r[i].second + diffReconst[i];
            }
        }
    }
    if (SECURITY_TYPE.compare("Malicious") == 0)
        funcCheckMaliciousDotProd(a, b, c, temp3, size);
}

// Term by term multiplication of 64-bit vectors overriding different precision
void funcDotProduct(const RSSVectorMyType &a, const RSSVectorMyType &b,
                    RSSVectorMyType &c, size_t size, bool truncation, vector<size_t> precision)
{
    log_print("funcDotProduct");
    assert(a.size() == size && "Matrix a incorrect for Mat-Mul");
    assert(b.size() == size && "Matrix b incorrect for Mat-Mul");
    assert(c.size() == size && "Matrix c incorrect for Mat-Mul");

    vector<myType> temp3(size, 0);

    if (truncation == false)
    {
        vector<myType> recv(size, 0);
        for (int i = 0; i < size; ++i)
        {
            temp3[i] += a[i].first * b[i].first +
                        a[i].first * b[i].second +
                        a[i].second * b[i].first;
        }

        thread *threads = new thread[2];

        threads[0] = thread(sendVector<myType>, ref(temp3), prevParty(partyNum), size);
        threads[1] = thread(receiveVector<myType>, ref(recv), nextParty(partyNum), size);

        for (int i = 0; i < 2; i++)
            threads[i].join();
        delete[] threads;

        for (int i = 0; i < size; ++i)
        {
            c[i].first = temp3[i];
            c[i].second = recv[i];
        }
    }
    else
    {
        vector<int> precisionInt(size, 0);
        for (int i = 0; i < size; ++i)
            precisionInt[i] = (1 << precision[i]);

        vector<myType> diffReconst(size, 0);
        RSSVectorMyType r(size), rPrime(size);
        PrecomputeObject.getDividedShares(r, rPrime, precisionInt, size);

        for (int i = 0; i < size; ++i)
        {
            temp3[i] += a[i].first * b[i].first +
                        a[i].first * b[i].second +
                        a[i].second * b[i].first -
                        rPrime[i].first;
        }

        funcReconstruct3out3(temp3, diffReconst, size, "Dot-product diff reconst", false);

        dividePlain(diffReconst, precisionInt);
        if (partyNum == PARTY_A)
        {
            for (int i = 0; i < size; ++i)
            {
                c[i].first = r[i].first + diffReconst[i];
                c[i].second = r[i].second;
            }
        }

        if (partyNum == PARTY_B)
        {
            for (int i = 0; i < size; ++i)
            {
                c[i].first = r[i].first;
                c[i].second = r[i].second;
            }
        }

        if (partyNum == PARTY_C)
        {
            for (int i = 0; i < size; ++i)
            {
                c[i].first = r[i].first;
                c[i].second = r[i].second + diffReconst[i];
            }
        }
    }
    if (SECURITY_TYPE.compare("Malicious") == 0)
        funcCheckMaliciousDotProd(a, b, c, temp3, size);
}

// Term by term multiplication of mod 67 vectors
void funcDotProduct(const RSSVectorSmallType &a, const RSSVectorSmallType &b,
                    RSSVectorSmallType &c, size_t size)
{
    log_print("funcDotProduct");
    assert(a.size() == size && "Matrix a incorrect for Mat-Mul");
    assert(b.size() == size && "Matrix b incorrect for Mat-Mul");
    assert(c.size() == size && "Matrix c incorrect for Mat-Mul");

    vector<smallType> temp3(size, 0), recv(size, 0);
    for (int i = 0; i < size; ++i)
    {
        temp3[i] = multiplicationModPrime[a[i].first][b[i].first];
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[a[i].first][b[i].second]];
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[a[i].second][b[i].first]];
    }

    // Add random shares of 0 locally
    thread *threads = new thread[2];

    threads[0] = thread(sendVector<smallType>, ref(temp3), prevParty(partyNum), size);
    threads[1] = thread(receiveVector<smallType>, ref(recv), nextParty(partyNum), size);

    for (int i = 0; i < 2; i++)
        threads[i].join();

    delete[] threads;

    for (int i = 0; i < size; ++i)
    {
        c[i].first = temp3[i];
        c[i].second = recv[i];
    }

    if (SECURITY_TYPE.compare("Malicious") == 0)
        funcCheckMaliciousDotProd(a, b, c, temp3, size);
}

// Term by term multiplication boolean shares
void funcDotProductBits(const RSSVectorSmallType &a, const RSSVectorSmallType &b,
                        RSSVectorSmallType &c, size_t size)
{
    log_print("funcDotProductBits");
    assert(a.size() == size && "Matrix a incorrect for Mat-Mul");
    assert(b.size() == size && "Matrix b incorrect for Mat-Mul");
    assert(c.size() == size && "Matrix c incorrect for Mat-Mul");

    vector<smallType> temp3(size, 0), recv(size, 0);
    for (int i = 0; i < size; ++i)
    {
        temp3[i] = (a[i].first and b[i].first) ^
                   (a[i].first and b[i].second) ^
                   (a[i].second and b[i].first);
    }

    // Add random shares of 0 locally
    thread *threads = new thread[2];
    threads[0] = thread(sendVector<smallType>, ref(temp3), prevParty(partyNum), size);
    threads[1] = thread(receiveVector<smallType>, ref(recv), nextParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
    {
        c[i].first = temp3[i];
        c[i].second = recv[i];
    }

    if (SECURITY_TYPE.compare("Malicious") == 0)
        funcCheckMaliciousDotProdBits(a, b, c, temp3, size);
}

// Multiply index 2i, 2i+1 of the first vector into the second one. The second vector is half the size.
void funcMultiplyNeighbours(const RSSVectorSmallType &c_1, RSSVectorSmallType &c_2, size_t size)
{
    assert(size % 2 == 0 && "Size should be 'half'able");
    vector<smallType> temp3(size / 2, 0), recv(size / 2, 0);
    for (int i = 0; i < size / 2; ++i)
    {
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2 * i].first][c_1[2 * i + 1].first]];
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2 * i].first][c_1[2 * i + 1].second]];
        temp3[i] = additionModPrime[temp3[i]][multiplicationModPrime[c_1[2 * i].second][c_1[2 * i + 1].first]];
    }

    // Add random shares of 0 locally
    thread *threads = new thread[2];

    threads[0] = thread(sendVector<smallType>, ref(temp3), nextParty(partyNum), size / 2);
    threads[1] = thread(receiveVector<smallType>, ref(recv), prevParty(partyNum), size / 2);

    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size / 2; ++i)
    {
        c_2[i].first = temp3[i];
        c_2[i].second = recv[i];
    }

    RSSVectorSmallType temp_a(size / 2), temp_b(size / 2), temp_c(size / 2);
    if (SECURITY_TYPE.compare("Malicious") == 0)
        funcCheckMaliciousDotProd(temp_a, temp_b, temp_c, temp3, size / 2);
}

// Multiply each group of 64 with a random number in Z_p* and reconstruct output in betaPrime.
void funcCrunchMultiply(const RSSVectorSmallType &c, vector<smallType> &betaPrime, size_t size)
{
    size_t sizeLong = size * BIT_SIZE;
    RSSVectorSmallType c_0(sizeLong / 2, make_pair(0, 0)), c_1(sizeLong / 4, make_pair(0, 0)),
        c_2(sizeLong / 8, make_pair(0, 0)), c_3(sizeLong / 16, make_pair(0, 0)),
        c_4(sizeLong / 32, make_pair(0, 0));
    RSSVectorSmallType c_5(sizeLong / 64, make_pair(0, 0));

    vector<smallType> reconst(size, 0);

    funcMultiplyNeighbours(c, c_0, sizeLong);
    funcMultiplyNeighbours(c_0, c_1, sizeLong / 2);
    funcMultiplyNeighbours(c_1, c_2, sizeLong / 4);
    funcMultiplyNeighbours(c_2, c_3, sizeLong / 8);
    funcMultiplyNeighbours(c_3, c_4, sizeLong / 16);
    if (BIT_SIZE == 64)
        funcMultiplyNeighbours(c_4, c_5, sizeLong / 32);

    vector<smallType> a_next(size), a_prev(size);
    if (BIT_SIZE == 64)
        for (int i = 0; i < size; ++i)
        {
            a_prev[i] = 0;
            a_next[i] = c_5[i].first;
            reconst[i] = c_5[i].first;
            reconst[i] = additionModPrime[reconst[i]][c_5[i].second];
        }
    else if (BIT_SIZE == 32)
        for (int i = 0; i < size; ++i)
        {
            a_prev[i] = 0;
            a_next[i] = c_4[i].first;
            reconst[i] = c_4[i].first;
            reconst[i] = additionModPrime[reconst[i]][c_4[i].second];
        }

    thread *threads = new thread[2];

    threads[0] = thread(sendVector<smallType>, ref(a_next), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<smallType>, ref(a_prev), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
        reconst[i] = additionModPrime[reconst[i]][a_prev[i]];

    for (int i = 0; i < size; ++i)
    {
        if (reconst[i] == 0)
            betaPrime[i] = 1;
    }
}

// Thread function for parallel private compare
void parallelFirst(smallType *temp3, const RSSSmallType *beta, const myType *r,
                   const RSSSmallType *share_m, size_t start, size_t end, int t)
{
    size_t index3, index2;
    smallType bit_r;
    RSSSmallType twoBetaMinusOne, diff;

    for (int index2 = start; index2 < end; ++index2)
    {
        // Computing 2Beta-1
        twoBetaMinusOne = subConstModPrime(beta[index2], 1);
        twoBetaMinusOne = addModPrime(twoBetaMinusOne, beta[index2]);

        for (size_t k = 0; k < BIT_SIZE; ++k)
        {
            index3 = index2 * BIT_SIZE + k;
            bit_r = (smallType)((r[index2] >> (BIT_SIZE - 1 - k)) & 1);
            diff = share_m[index3];

            if (bit_r == 1)
                diff = subConstModPrime(diff, 1);

            // Dot Product
            temp3[index3] = multiplicationModPrime[diff.first][twoBetaMinusOne.first];
            temp3[index3] = additionModPrime[temp3[index3]][multiplicationModPrime[diff.first][twoBetaMinusOne.second]];
            temp3[index3] = additionModPrime[temp3[index3]][multiplicationModPrime[diff.second][twoBetaMinusOne.first]];
        }
    }
}

void parallelSecond(RSSSmallType *c, const smallType *temp3, const smallType *recv, const myType *r,
                    const RSSSmallType *share_m, size_t start, size_t end, int t)
{
    size_t index3, index2;
    smallType bit_r;
    RSSSmallType a, tempM, tempN, xMinusR;

    if (partyNum == PARTY_A)
    {
        for (int index2 = start; index2 < end; ++index2)
        {
            a = make_pair(0, 0);
            for (size_t k = 0; k < BIT_SIZE; ++k)
            {
                index3 = index2 * BIT_SIZE + k;
                // Complete Dot Product
                xMinusR.first = temp3[index3];
                xMinusR.second = recv[index3];

                // Resume rest of the loop
                c[index3] = a;
                tempM = share_m[index3];
                bit_r = (smallType)((r[index2] >> (BIT_SIZE - 1 - k)) & 1);

                tempN = XORPublicModPrime(tempM, bit_r);
                a = addModPrime(a, tempN);

                c[index3].first = additionModPrime[c[index3].first][xMinusR.first];
                c[index3].first = additionModPrime[c[index3].first][1];
                c[index3].second = additionModPrime[c[index3].second][xMinusR.second];
            }
        }
    }

    if (partyNum == PARTY_B)
    {
        for (int index2 = start; index2 < end; ++index2)
        {
            a = make_pair(0, 0);
            for (size_t k = 0; k < BIT_SIZE; ++k)
            {
                index3 = index2 * BIT_SIZE + k;
                // Complete Dot Product
                xMinusR.first = temp3[index3];
                xMinusR.second = recv[index3];

                // Resume rest of the loop
                c[index3] = a;
                tempM = share_m[index3];
                bit_r = (smallType)((r[index2] >> (BIT_SIZE - 1 - k)) & 1);

                tempN = XORPublicModPrime(tempM, bit_r);
                a = addModPrime(a, tempN);

                c[index3].first = additionModPrime[c[index3].first][xMinusR.first];
                c[index3].second = additionModPrime[c[index3].second][xMinusR.second];
            }
        }
    }

    if (partyNum == PARTY_C)
    {
        for (int index2 = start; index2 < end; ++index2)
        {
            a = make_pair(0, 0);
            for (size_t k = 0; k < BIT_SIZE; ++k)
            {
                index3 = index2 * BIT_SIZE + k;
                // Complete Dot Product
                xMinusR.first = temp3[index3];
                xMinusR.second = recv[index3];

                // Resume rest of the loop
                c[index3] = a;
                tempM = share_m[index3];
                bit_r = (smallType)((r[index2] >> (BIT_SIZE - 1 - k)) & 1);

                tempN = XORPublicModPrime(tempM, bit_r);
                a = addModPrime(a, tempN);

                c[index3].first = additionModPrime[c[index3].first][xMinusR.first];
                c[index3].second = additionModPrime[c[index3].second][xMinusR.second];
                c[index3].second = additionModPrime[c[index3].second][1];
            }
        }
    }
}

// Private Compare functionality
void funcPrivateCompare(const RSSVectorSmallType &share_m, const vector<myType> &r,
                        const RSSVectorSmallType &beta, vector<smallType> &betaPrime,
                        size_t size)
{
    log_print("funcPrivateCompare");
    assert(share_m.size() == size * BIT_SIZE && "Input error share_m");
    assert(r.size() == size && "Input error r");
    assert(beta.size() == size && "Input error beta");

    size_t sizeLong = size * BIT_SIZE;
    size_t index3, index2;
    RSSVectorSmallType c(sizeLong), diff(sizeLong), twoBetaMinusOne(sizeLong), xMinusR(sizeLong);
    RSSSmallType a, tempM, tempN;
    smallType bit_r;

    // Computing x[i] - r[i]
    if (PARALLEL)
    {
        assert(NO_CORES > 2 && "Need at least 2 cores for threads variable abuse");
        vector<smallType> temp3(sizeLong, 0), recv(sizeLong, 0);

        // First part of parallel execution
        thread *threads = new thread[NO_CORES];
        int chunksize = size / NO_CORES;

        for (int i = 0; i < NO_CORES; i++)
        {
            int start = i * chunksize;
            int end = (i + 1) * chunksize;
            if (i == NO_CORES - 1)
                end = size;

            threads[i] = thread(parallelFirst, temp3.data(), beta.data(), r.data(),
                                share_m.data(), start, end, i);
        }
        for (int i = 0; i < NO_CORES; i++)
            threads[i].join();

        //"Single" threaded execution
        threads[0] = thread(sendVector<smallType>, ref(temp3), prevParty(partyNum), size);
        threads[1] = thread(receiveVector<smallType>, ref(recv), nextParty(partyNum), size);

        for (int i = 0; i < 2; i++)
            threads[i].join();

        // Parallel execution resumes
        for (int i = 0; i < NO_CORES; i++)
        {
            int start = i * chunksize;
            int end = (i + 1) * chunksize;
            if (i == NO_CORES - 1)
                end = size;

            threads[i] = thread(parallelSecond, c.data(), temp3.data(), recv.data(),
                                r.data(), share_m.data(), start, end, i);
        }

        for (int i = 0; i < NO_CORES; i++)
            threads[i].join();
        delete[] threads;
    }
    else
    {
        for (int index2 = 0; index2 < size; ++index2)
        {
            // Computing 2Beta-1
            twoBetaMinusOne[index2 * BIT_SIZE] = subConstModPrime(beta[index2], 1);
            twoBetaMinusOne[index2 * BIT_SIZE] = addModPrime(twoBetaMinusOne[index2 * BIT_SIZE], beta[index2]);

            for (size_t k = 0; k < BIT_SIZE; ++k)
            {
                index3 = index2 * BIT_SIZE + k;
                twoBetaMinusOne[index3] = twoBetaMinusOne[index2 * BIT_SIZE];

                bit_r = (smallType)((r[index2] >> (BIT_SIZE - 1 - k)) & 1);
                diff[index3] = share_m[index3];

                if (bit_r == 1)
                    diff[index3] = subConstModPrime(diff[index3], 1);
            }
        }

        //(-1)^beta * x[i] - r[i]
        funcDotProduct(diff, twoBetaMinusOne, xMinusR, sizeLong);

        for (int index2 = 0; index2 < size; ++index2)
        {
            a = make_pair(0, 0);
            for (size_t k = 0; k < BIT_SIZE; ++k)
            {
                index3 = index2 * BIT_SIZE + k;
                c[index3] = a;
                tempM = share_m[index3];

                bit_r = (smallType)((r[index2] >> (BIT_SIZE - 1 - k)) & 1);

                tempN = XORPublicModPrime(tempM, bit_r);
                a = addModPrime(a, tempN);

                if (partyNum == PARTY_A)
                {
                    c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
                    c[index3].first = additionModPrime[c[index3].first][1];
                    c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
                }
                else if (partyNum == PARTY_B)
                {
                    c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
                    c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
                }
                else if (partyNum == PARTY_C)
                {
                    c[index3].first = additionModPrime[c[index3].first][xMinusR[index3].first];
                    c[index3].second = additionModPrime[c[index3].second][xMinusR[index3].second];
                    c[index3].second = additionModPrime[c[index3].second][1];
                }
            }
        }
    }

    RSSVectorSmallType temp_a(sizeLong / 2), temp_b(sizeLong / 2), temp_c(sizeLong / 2);
    vector<smallType> temp_d(sizeLong / 2);
    if (SECURITY_TYPE.compare("Malicious") == 0)
        funcCheckMaliciousDotProd(temp_a, temp_b, temp_c, temp_d, sizeLong / 2);

    // TODO 7 rounds of multiplication
    //  cout << "CM: \t\t" << funcTime(funcCrunchMultiply, c, betaPrime, size) << endl;
    funcCrunchMultiply(c, betaPrime, size);
}

// Wrap functionality.
void funcWrap(const RSSVectorMyType &a, RSSVectorSmallType &theta, size_t size)
{
    log_print("funcWrap");

    size_t sizeLong = size * BIT_SIZE;
    RSSVectorMyType x(size), r(size);
    RSSVectorSmallType shares_r(sizeLong), alpha(size), beta(size), eta(size);
    vector<smallType> delta(size), etaPrime(size);
    vector<myType> reconst_x(size);

    PrecomputeObject.getShareConvertObjects(r, shares_r, alpha, size);
    addVectors<RSSMyType>(a, r, x, size);
    for (int i = 0; i < size; ++i)
    {
        beta[i].first = wrapAround(a[i].first, r[i].first);
        x[i].first = a[i].first + r[i].first;
        beta[i].second = wrapAround(a[i].second, r[i].second);
        x[i].second = a[i].second + r[i].second;
    }

    vector<myType> x_next(size), x_prev(size);
    for (int i = 0; i < size; ++i)
    {
        x_prev[i] = 0;
        x_next[i] = x[i].first;
        reconst_x[i] = x[i].first;
        reconst_x[i] = reconst_x[i] + x[i].second;
    }

    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(x_next), nextParty(partyNum), size);
    threads[1] = thread(receiveVector<myType>, ref(x_prev), prevParty(partyNum), size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    for (int i = 0; i < size; ++i)
        reconst_x[i] = reconst_x[i] + x_prev[i];

    wrap3(x, x_prev, delta, size); // All parties have delta
    PrecomputeObject.getRandomBitShares(eta, size);

    // cout << "PC: \t\t" << funcTime(funcPrivateCompare, shares_r, reconst_x, eta, etaPrime, size, BIT_SIZE) << endl;
    funcPrivateCompare(shares_r, reconst_x, eta, etaPrime, size);

    if (partyNum == PARTY_A)
    {
        for (int i = 0; i < size; ++i)
        {
            theta[i].first = beta[i].first ^ delta[i] ^ alpha[i].first ^ eta[i].first ^ etaPrime[i];
            theta[i].second = beta[i].second ^ alpha[i].second ^ eta[i].second;
        }
    }
    else if (partyNum == PARTY_B)
    {
        for (int i = 0; i < size; ++i)
        {
            theta[i].first = beta[i].first ^ delta[i] ^ alpha[i].first ^ eta[i].first;
            theta[i].second = beta[i].second ^ alpha[i].second ^ eta[i].second;
        }
    }
    else if (partyNum == PARTY_C)
    {
        for (int i = 0; i < size; ++i)
        {
            theta[i].first = beta[i].first ^ alpha[i].first ^ eta[i].first;
            theta[i].second = beta[i].second ^ delta[i] ^ alpha[i].second ^ eta[i].second ^ etaPrime[i];
        }
    }
}

// Set c[i] = a[i] if b[i] = 0
// Set c[i] = 0    if b[i] = 1
void funcSelectShares(const RSSVectorMyType &a, const RSSVectorSmallType &b,
                      RSSVectorMyType &selected, size_t size)
{
    log_print("funcSelectShares");

    RSSVectorSmallType c(size), bXORc(size);
    RSSVectorMyType m_c(size);
    vector<smallType> reconst_b(size);
    PrecomputeObject.getSelectorBitShares(c, m_c, size);

    for (int i = 0; i < size; ++i)
    {
        bXORc[i].first = c[i].first ^ b[i].first;
        bXORc[i].second = c[i].second ^ b[i].second;
    }

    funcReconstructBit(bXORc, reconst_b, size, "bXORc", false);

    if (partyNum == PARTY_A)
        for (int i = 0; i < size; ++i)
            if (reconst_b[i] == 0)
            {
                m_c[i].first = (myType)1 - m_c[i].first;
                m_c[i].second = -m_c[i].second;
            }

    if (partyNum == PARTY_B)
        for (int i = 0; i < size; ++i)
            if (reconst_b[i] == 0)
            {
                m_c[i].first = -m_c[i].first;
                m_c[i].second = -m_c[i].second;
            }

    if (partyNum == PARTY_C)
        for (int i = 0; i < size; ++i)
            if (reconst_b[i] == 0)
            {
                m_c[i].first = -m_c[i].first;
                m_c[i].second = (myType)1 - m_c[i].second;
            }

    funcDotProduct(a, m_c, selected, size, false, 0);
}

// Within each group of columns, select a0 or a1 depending on value of bit b into answer.
// loopCounter is used to offset a1 by loopCounter*rows*columns
// answer = ((a0 \oplus a1) b ) \oplus a0
void funcSelectBitShares(const RSSVectorSmallType &a0, const RSSVectorSmallType &a1,
                         const RSSVectorSmallType &b, RSSVectorSmallType &answer,
                         size_t rows, size_t columns, size_t loopCounter)
{
    log_print("funcSelectBitShares");
    size_t size = rows * columns;
    assert(a0.size() == rows * columns && "a0 size incorrect");
    assert(a1.size() == (columns)*rows * columns && "a1 size incorrect");
    assert(b.size() == rows && "b size incorrect");
    assert(answer.size() == rows * columns && "answers size incorrect");

    RSSVectorSmallType bRepeated(size), tempXOR(size);
    for (int i = 0; i < rows; ++i)
        for (size_t j = 0; j < columns; ++j)
            bRepeated[i * columns + j] = b[i];

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < columns; ++j)
            tempXOR[i * columns + j] = a0[i * columns + j] ^
                                       a1[loopCounter * rows * columns + i * columns + j];

    funcDotProductBits(tempXOR, bRepeated, answer, size);

    for (int i = 0; i < size; ++i)
        answer[i] = answer[i] ^ a0[i];
}

// b holds bits of ReLU' of a.
// Warning: when a <= 0, b = 0, else b = 1.
void funcRELUPrime(const RSSVectorMyType &a, RSSVectorSmallType &b, size_t size)
{
    log_print("funcRELUPrime");

    RSSVectorMyType twoA(size);
    RSSVectorSmallType theta(size);
    for (int i = 0; i < size; ++i)
        twoA[i] = a[i] << 1;

    // cout << "Wrap: \t\t" << funcTime(funcWrap, twoA, theta, size) << endl;
    funcWrap(twoA, theta, size);

    for (int i = 0; i < size; ++i)
    {
        b[i].first = theta[i].first ^ (getMSB(a[i].first));
        b[i].second = theta[i].second ^ (getMSB(a[i].second));
    }

    // 1 XOR b
    // if (partyNum == PARTY_A)
    //     for (int i = 0; i < size; ++i)
    //         b[i].first = 1 ^ b[i].first;
    // if (partyNum == PARTY_B)
    //     for (int i = 0; i < size; ++i)
    //         b[i].second = 1 ^ b[i].second;
}

// Input is a, outputs are temp = ReLU'(a) and b = RELU(a).
void funcRELU(const RSSVectorMyType &a, RSSVectorSmallType &temp, RSSVectorMyType &b, size_t size)
{
    log_print("funcRELU");

    RSSVectorSmallType c(size), bXORc(size);
    RSSVectorMyType m_c(size);
    vector<smallType> reconst_b(size);

    // cout << "ReLU': \t\t" << funcTime(funcRELUPrime, a, temp, size) << endl;
    // if size is too large, split into batches
    if (size > BATCH_COMPUTE_THRESHOLD_RELU)
    {
        RSSVectorMyType a_batch(BATCH_COMPUTE_THRESHOLD_RELU);
        RSSVectorSmallType temp_batch(BATCH_COMPUTE_THRESHOLD_RELU);
        size_t batch_num = size / BATCH_COMPUTE_THRESHOLD_RELU;
        for (size_t i = 0; i < batch_num; i++)
        {
            copyVectors<RSSMyType>(a, a_batch, i * BATCH_COMPUTE_THRESHOLD_RELU, BATCH_COMPUTE_THRESHOLD_RELU);
            funcRELUPrime(a_batch, temp_batch, BATCH_COMPUTE_THRESHOLD_RELU);
            for (size_t j = 0; j < BATCH_COMPUTE_THRESHOLD_RELU; j++)
                temp[i * BATCH_COMPUTE_THRESHOLD_RELU + j] = temp_batch[j];
        }

        if (size % BATCH_COMPUTE_THRESHOLD_RELU != 0)
        {
            a_batch.resize(size % BATCH_COMPUTE_THRESHOLD_RELU);
            temp_batch.resize(size % BATCH_COMPUTE_THRESHOLD_RELU);
            copyVectors<RSSMyType>(a, a_batch, batch_num * BATCH_COMPUTE_THRESHOLD_RELU, size % BATCH_COMPUTE_THRESHOLD_RELU);
            funcRELUPrime(a_batch, temp_batch, size % BATCH_COMPUTE_THRESHOLD_RELU);
            for (size_t j = 0; j < size % BATCH_COMPUTE_THRESHOLD_RELU; j++)
                temp[batch_num * BATCH_COMPUTE_THRESHOLD_RELU + j] = temp_batch[j];
        }
    }
    else
        funcRELUPrime(a, temp, size);
    PrecomputeObject.getSelectorBitShares(c, m_c, size);

    for (int i = 0; i < size; ++i)
    {
        bXORc[i].first = c[i].first ^ temp[i].first;
        bXORc[i].second = c[i].second ^ temp[i].second;
    }

    funcReconstructBit(bXORc, reconst_b, size, "bXORc", false);
    if (partyNum == PARTY_A)
        for (int i = 0; i < size; ++i)
            if (reconst_b[i] == 0)
            {
                m_c[i].first = (myType)1 - m_c[i].first;
                m_c[i].second = -m_c[i].second;
            }

    if (partyNum == PARTY_B)
        for (int i = 0; i < size; ++i)
            if (reconst_b[i] == 0)
            {
                m_c[i].first = -m_c[i].first;
                m_c[i].second = -m_c[i].second;
            }

    if (partyNum == PARTY_C)
        for (int i = 0; i < size; ++i)
            if (reconst_b[i] == 0)
            {
                m_c[i].first = -m_c[i].first;
                m_c[i].second = (myType)1 - m_c[i].second;
            }

    // vector<myType> reconst_m_c(size);
    // funcReconstruct(m_c, reconst_m_c, size, "m_c", true);
    funcDotProduct(a, m_c, b, size, false, 0);
}

void funcPow(const RSSVectorMyType &b, vector<smallType> &alpha, size_t size)
{
    size_t ell = 5;
    if (BIT_SIZE == 64)
        ell = 6;

    RSSVectorMyType x(size), d(size), temp(size);
    copyVectors<RSSMyType>(b, x, size);

    RSSVectorSmallType c(size);
    for (int i = 0; i < size; ++i)
        alpha[i] = 0;

    vector<smallType> r_c(size);
    vector<myType> temp_1(size);
    for (int j = ell - 1; j > -1; --j)
    {
        for (size_t i = 0; i < size; ++i)
            temp_1[i] = ((myType)1 << ((int)(1 << j) + (int)alpha[i]));

        funcGetShares(temp, temp_1);
        subtractVectors<RSSMyType>(x, temp, d, size);
        funcRELUPrime(d, c, size);
        funcReconstructBit(c, r_c, size, "null", false);
        for (int i = 0; i < size; ++i)
            if (r_c[i] == 0)
                alpha[i] += (1 << j);
    }
}

// All parties start with shares of a number in a and b and the quotient is in quotient.
// alpha is the order of divisiors, 2^alpha =< b < 2^{alpha+1}.
void funcDivision(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &quotient,
                  size_t size)
{
    log_print("funcDivision");

    vector<smallType> alpha_temp(size);
    funcPow(b, alpha_temp, size);

    vector<size_t> alpha(size);
    vector<size_t> precision(size);
    for (int i = 0; i < size; ++i)
    {
        alpha[i] = alpha_temp[i];
        precision[i] = alpha[i] + 1;
    }
    vector<myType> data_twoPointNine(size), data_one(size), reconst(size);
    for (int i = 0; i < size; ++i)
    {
        data_twoPointNine[i] = ((myType)(2.9142 * (1 << precision[i])));
        data_one[i] = ((myType)(1 * (1 << precision[i])));
    }

    RSSVectorMyType ones(size), twoPointNine(size), twoX(size), w0(size), xw0(size),
        epsilon0(size), epsilon1(size), termOne(size), termTwo(size), answer(size);
    funcGetShares(twoPointNine, data_twoPointNine);
    funcGetShares(ones, data_one);

    multiplyByScalar(b, 2, twoX);
    subtractVectors<RSSMyType>(twoPointNine, twoX, w0, size);
    funcDotProduct(b, w0, xw0, size, true, precision);
    subtractVectors<RSSMyType>(ones, xw0, epsilon0, size);
    if (PRECISE_DIVISION)
        funcDotProduct(epsilon0, epsilon0, epsilon1, size, true, precision);
    addVectors(ones, epsilon0, termOne, size);
    if (PRECISE_DIVISION)
        addVectors(ones, epsilon1, termTwo, size);
    funcDotProduct(w0, termOne, answer, size, true, precision);
    if (PRECISE_DIVISION)
        funcDotProduct(answer, termTwo, answer, size, true, precision);

    // RSSVectorMyType scaledA(size);
    // vector<size_t> alphaPlusOne(size);
    // for (int i = 0; i < size; ++i)
    //     alphaPlusOne[i] = (1 << (alpha[i] + 1));
    // multiplyByScalar(a, alphaPlusOne, scaledA);

    vector<size_t> precisionPrime(size);
    for (int i = 0; i < size; ++i)
        precisionPrime[i] = 2 * precision[i] - FLOAT_PRECISION;
    funcDotProduct(answer, a, quotient, size, true, precisionPrime);
}

// Chunk wise maximum of a vector of size rows*columns and maximum is caclulated of every
// column number of elements. max is a vector of size rows, maxPrime, of rows*columns*columns;
void funcMaxpool(RSSVectorMyType &a, RSSVectorMyType &max, RSSVectorSmallType &maxPrime,
                 size_t rows, size_t columns)
{
    log_print("funcMaxpool");
    assert(columns < 256 && "Pooling size has to be smaller than 8-bits");

    size_t size = rows * columns;
    RSSVectorMyType diff(rows);
    RSSVectorSmallType rp(rows), dmpIndexShares(columns * size), temp(size);
    vector<smallType> dmpTemp(columns * size, 0);

    for (int loopCounter = 0; loopCounter < columns; ++loopCounter)
        for (size_t i = 0; i < rows; ++i)
            dmpTemp[loopCounter * rows * columns + i * columns + loopCounter] = 1;
    funcGetShares(dmpIndexShares, dmpTemp);

    for (size_t i = 0; i < size; ++i)
        maxPrime[i] = dmpIndexShares[i];

    for (size_t i = 0; i < rows; ++i)
        max[i] = a[i * columns];

    for (size_t i = 1; i < columns; ++i)
    {
        for (size_t j = 0; j < rows; ++j)
            diff[j] = max[j] - a[j * columns + i];

        funcRELU(diff, rp, max, rows);
        funcSelectBitShares(maxPrime, dmpIndexShares, rp, temp, rows, columns, i);

        for (size_t i = 0; i < size; ++i)
            maxPrime[i] = temp[i];

        for (size_t j = 0; j < rows; ++j)
            max[j] = max[j] + a[j * columns + i];
    }
}

void funcInvSqrt(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size)
{
    log_print("funcInvSqrt");

    // invoke funcPow
    vector<smallType> alpha(size);
    funcPow(a, alpha, size);

    // compute 2^{-alpha/2}
    vector<myType> twoAlphaHalf(size);
    double result = 0;
    for (int i = 0; i < size; ++i)
    {
        // compute 2^{-alpha/2} = 1 / sqrt(2^alpha)
        if (BIT_SIZE == 32)
            result = (static_cast<int32_t>(1 << (int)alpha[i])) / (double)(1 << FLOAT_PRECISION);
        else if (BIT_SIZE == 64)
            result = (static_cast<int64_t>(1 << (int)alpha[i])) / (double)(1 << FLOAT_PRECISION);
        result = 1.0 / sqrt(result);
        twoAlphaHalf[i] = floatToMyType(result);
    }

    vector<RSSVectorMyType> y(INV_SQRT_ITER_NUM + 1, RSSVectorMyType(size));
    funcGetShares(y[0], twoAlphaHalf);

    vector<myType> data_Three(size, floatToMyType(3));

    RSSVectorMyType three(size);
    funcGetShares(three, data_Three);

    RSSVectorMyType gamma(size), xGamma(size), delta(size), yDelta(size);
    for (int i = 0; i < INV_SQRT_ITER_NUM; ++i)
    {
        funcDotProduct(y[i], y[i], gamma, size, true, FLOAT_PRECISION);
        funcDotProduct(a, gamma, xGamma, size, true, FLOAT_PRECISION);
        subtractVectors<RSSMyType>(three, xGamma, delta, size);
        funcDotProduct(y[i], delta, yDelta, size, true, FLOAT_PRECISION);
        // compute 1/2 * y[i] * delta
        funcTruncatePublic(yDelta, 2, size);
        y[i + 1] = yDelta;
    }
    // set b = y[INV_SQRT_ITER_NUM]
    b.assign(y[INV_SQRT_ITER_NUM].begin(), y[INV_SQRT_ITER_NUM].end());
}

void funcInv(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size)
{
    log_print("funcInv");
    assert(a.size() == size && "Vector a incorrect for Inv");
    assert(b.size() == size && "Vector b incorrect for Inv");

    // invoke funcDivision
    RSSVectorMyType ones(size);
    vector<myType> data_one(size, floatToMyType(1));
    funcGetShares(ones, data_one);
    funcDivision(ones, a, b, size);
}

void funcInnerProduct(const RSSVectorMyType &a, const RSSVectorMyType &b,
                      RSSMyType &c, size_t size, size_t precision)
{
    log_print("funcInnerProduct");
    assert(a.size() == size && "Matrix a incorrect for MatMul");
    assert(b.size() == size && "Matrix b incorrect for MatMul");

    // compute a * b
    RSSVectorMyType tempProd(size);
    funcDotProduct(a, b, tempProd, size, true, precision);

    // compute inner product
    for (size_t i = 0; i < size; ++i)
    {
        c.first += tempProd[i].first;
        c.second += tempProd[i].second;
    }
}

void funcSoftmax(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size)
{
    log_print("funcSoftmax");

    RSSVectorMyType temp(size);
    copyVectors<RSSMyType>(a, temp, size);

    // compute 1/t * a
    funcTruncate(temp, floor(log2(SOFTMAX_ITER_NUM)), size);

    vector<RSSVectorMyType> f(SOFTMAX_ITER_NUM + 1, RSSVectorMyType(size));
    // initialize f[0]
    myType initData = floatToMyType(1.0 / size);
    funcGetShares(f[0], vector<myType>(size, initData));

    // compute f[j-1/t]
    for (size_t j = 1; j <= SOFTMAX_ITER_NUM; ++j)
    {
        // compute inner product, x = temp * f[j-1/t]
        RSSMyType x;
        funcInnerProduct(temp, f[j - 1], x, size, FLOAT_PRECISION);
        RSSVectorMyType xVector(size, x);
        // compute b = x*f[j-1/t]， compute c = temp * f[j-1/t]
        RSSVectorMyType bVector(size), cVector(size);
        funcDotProduct(xVector, f[j - 1], bVector, size, true, FLOAT_PRECISION);
        funcDotProduct(temp, f[j - 1], cVector, size, true, FLOAT_PRECISION);
        // compute f[j] = f[j-1/t] - b + c
        for (size_t i = 0; i < size; ++i)
        {
            f[j][i].first = f[j - 1][i].first - bVector[i].first + cVector[i].first;
            f[j][i].second = f[j - 1][i].second - bVector[i].second + cVector[i].second;
        }
    }
    // compute b = f[t/t]
    b.assign(f[SOFTMAX_ITER_NUM].begin(), f[SOFTMAX_ITER_NUM].end());
}

void funcArrayAccess(const RSSVectorMyType &a, const RSSMyType index, RSSMyType &b, size_t size)
{
    log_print("funcArrayAccess");
    assert(a.size() == size && "Array a incorrect for ArrayAccess");

    // rotate a
    vector<myType> tempA(size);
    RSSMyType tempIndex;

    // myType r_indep = aes_indep->get64Bits() % size;
    myType r_next = aes_next->get64Bits() % size;
    myType r_prev = aes_prev->get64Bits() % size;

    for (size_t i = 0; i < size; ++i)
    {
        tempA[(i + r_next + r_prev) % size] = a[i].first;
    }

    // offset tempIndex, note that [tempIndex + r_a]
    tempIndex.first = index.first + r_prev;
    tempIndex.second = index.second + r_next;

    // send tempA and tempIndex.first to previous party
    vector<myType> tempIndexVector(1, tempIndex.first);
    // receive tempIndexPrev and tempAPrev
    vector<myType> tempAPrev(size), tempIndexPrevVector(1);
    thread *threads = new thread[2];
    threads[0] = thread(sendTwoVectors<myType>, ref(tempA), ref(tempIndexVector), nextParty(partyNum), size, 1);
    threads[1] = thread(receiveTwoVectors<myType>, ref(tempAPrev), ref(tempIndexPrevVector), prevParty(partyNum), size, 1);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    // reconstruct h
    myType hMyType = tempIndexPrevVector[0] + tempIndex.first + tempIndex.second;
    size_t h = hMyType % size;

    // get a[h]
    if (h > size - 1)
        error("index is out of range!");

    // rotate tempAPrev
    vector<myType> tempAPrevRotated(size);
    for (size_t i = 0; i < size; ++i)
    {
        tempAPrevRotated[(i + r_next) % size] = tempAPrev[i];
    }

    // compute a[h] = tempAPrevRotated[h]
    // randomize the value of a[h], local generate t, where t_0 + t_1 + t_2 = 0
    myType t = aes_prev->get64Bits() - aes_next->get64Bits();
    myType a_h = tempAPrevRotated[h] + t;

    // send a_h to prev party
    thread *threads2 = new thread[2];
    vector<myType> a_h_vector(1, a_h), b_vector(1);
    threads2[0] = thread(sendVector<myType>, ref(a_h_vector), prevParty(partyNum), 1);
    threads2[1] = thread(receiveVector<myType>, ref(b_vector), nextParty(partyNum), 1);
    for (int i = 0; i < 2; i++)
        threads2[i].join();
    delete[] threads2;

    // set b.first = a[h], b.second = next party's a[h]
    b.first = a_h;
    b.second = b_vector[0];
}

void funcArrayAccessVector(const vector<RSSVectorMyType> &a, const RSSMyType index, RSSVectorMyType &b,
                           size_t size, size_t element_size)
{
    log_print("funcArrayAccessVector");
    assert(a.size() == size && "Array a incorrect for Array Access Vector");
    assert(b.size() == element_size && "Array b incorrect for Array Access Vector");

    // rotate a
    vector<myType> tempA(size * element_size);
    RSSMyType tempIndex;

    // myType r_indep = aes_indep->get64Bits() % size;
    myType r_next = aes_next->get64Bits() % size;
    myType r_prev = aes_prev->get64Bits() % size;

    for (size_t i = 0; i < size; ++i)
    {
        size_t offset = (i + r_next + r_prev) % size;
        for (size_t j = 0; j < element_size; ++j)
            tempA[offset * element_size + j] = a[i][j].first;
    }

    // offset tempIndex, note that [tempIndex + r_a]
    tempIndex.first = index.first + r_prev;
    tempIndex.second = index.second + r_next;

    // send tempA and tempIndex.first to previous party
    vector<myType> tempIndexVector(1, tempIndex.first);
    // receive tempIndexPrev and tempAPrev
    vector<myType> tempAPrev(size * element_size), tempIndexPrevVector(1);
    thread *threads = new thread[2];
    threads[0] = thread(sendTwoVectors<myType>, ref(tempA), ref(tempIndexVector), nextParty(partyNum), size * element_size, 1);
    threads[1] = thread(receiveTwoVectors<myType>, ref(tempAPrev), ref(tempIndexPrevVector), prevParty(partyNum), size * element_size, 1);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    // reconstruct h
    myType hMyType = tempIndexPrevVector[0] + tempIndex.first + tempIndex.second;
    size_t h = hMyType % size;

    // get a[h]
    if (h > size - 1)
        error("index is out of range!");

    // rotate tempAPrev
    vector<myType> tempAPrevRotated(size * element_size);
    for (size_t i = 0; i < size; ++i)
    {
        size_t offset = (i + r_next) % size;
        for (size_t j = 0; j < element_size; ++j)
            tempAPrevRotated[offset * element_size + j] = tempAPrev[i * element_size + j];
    }

    // compute a[h]
    vector<myType> a_h_vector(element_size), b_vector(element_size);
    myType t;
    for (size_t j = 0; j < element_size; ++j)
    {
        // randomize the value of a[h], local generate t, where t_0 + t_1 + t_2 = 0
        t = aes_prev->get64Bits() - aes_next->get64Bits();
        a_h_vector[j] = tempAPrevRotated[h * element_size + j] + t;
    }

    // send a_h to prev party
    thread *threads2 = new thread[2];
    threads2[0] = thread(sendVector<myType>, ref(a_h_vector), prevParty(partyNum), element_size);
    threads2[1] = thread(receiveVector<myType>, ref(b_vector), nextParty(partyNum), element_size);
    for (int i = 0; i < 2; i++)
        threads2[i].join();
    delete[] threads2;
    // set b.first = a[h], b.second = next party's a[h]
    for (size_t j = 0; j < element_size; ++j)
    {
        b[j].first = a_h_vector[j];
        b[j].second = b_vector[j];
    }
}

// a is of size batchSize*B, b is of size B and quotient is a/b (b from each group).
// Used in BNLayer.
void funcBatchNorm(const RSSVectorMyType &a, const RSSVectorMyType &b, RSSVectorMyType &quotient,
                   size_t batchSize, size_t B)
{
    log_print("funcBatchNorm");
    assert(a.size() == batchSize * B && "funcBatchNorm a size incorrect");
    assert(b.size() == B && "funcBatchNorm b size incorrect");
    assert(quotient.size() == batchSize * B && "funcBatchNorm quotient size incorrect");

    vector<smallType> alpha_temp(B);
    funcPow(b, alpha_temp, B);

    vector<size_t> alpha(B), precision(B);
    for (int i = 0; i < B; ++i)
    {
        alpha[i] = alpha_temp[i];
        precision[i] = alpha[i] + 1;
    }

    vector<myType> data_twoPointNine(B), data_one(B), reconst(B);
    for (int i = 0; i < B; ++i)
    {
        data_twoPointNine[i] = ((myType)(2.9142 * (1 << precision[i])));
        data_one[i] = ((myType)(1 * (1 << precision[i])));
    }

    RSSVectorMyType ones(B), twoPointNine(B), twoX(B), w0(B), xw0(B),
        epsilon0(B), epsilon1(B), termOne(B), termTwo(B), answer(B);
    funcGetShares(twoPointNine, data_twoPointNine);
    funcGetShares(ones, data_one);

    multiplyByScalar(b, 2, twoX);
    subtractVectors<RSSMyType>(twoPointNine, twoX, w0, B);
    funcDotProduct(b, w0, xw0, B, true, precision);
    subtractVectors<RSSMyType>(ones, xw0, epsilon0, B);
    if (PRECISE_DIVISION)
        funcDotProduct(epsilon0, epsilon0, epsilon1, B, true, precision);
    addVectors(ones, epsilon0, termOne, B);
    if (PRECISE_DIVISION)
        addVectors(ones, epsilon1, termTwo, B);
    funcDotProduct(w0, termOne, answer, B, true, precision);
    if (PRECISE_DIVISION)
        funcDotProduct(answer, termTwo, answer, B, true, precision);

    RSSVectorMyType scaledA(batchSize * B), b_repeat(batchSize * B);
    vector<size_t> precisionPrime(batchSize * B);
    // multiplyByScalar(a, 2, scaledA); //So that a is of precision precision
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < batchSize; ++j)
        {
            b_repeat[i * batchSize + j] = answer[i];
            precisionPrime[i * batchSize + j] = 2 * precision[i] - FLOAT_PRECISION;
        }

    funcDotProduct(b_repeat, a, quotient, batchSize * B, true, precisionPrime); // Convert to fixed precision
}

void funcForbeniusNorm(const RSSVectorMyType &a, RSSVectorMyType &b, size_t size)
{
    log_print("funcForbeniusNorm");
    assert(a.size() == size && "funcForbeniusNorm a size incorrect");

    // compute a * a and sum up
    RSSMyType temp;
    funcInnerProduct(a, a, temp, size, FLOAT_PRECISION);

    // compute 1 / sqrt(temp)
    RSSVectorMyType tempVector(1), result(1);
    tempVector[0] = temp;
    funcInvSqrt(tempVector, result, 1);

    // compute b = a * 1 / sqrt(temp)
    RSSVectorMyType resultVector(size, result[0]);
    funcDotProduct(a, resultVector, b, size, true, FLOAT_PRECISION);
}

void funcGenerateBaseArray(const RSSVectorMyType &a, vector<vector<myType>> &base_array, RSSVectorMyType &base_r, size_t base_array_size, size_t element_size)
{
    log_print("funcGenerateBaseArray");
    assert(a.size() == element_size && "Array a incorrect for GenerateBaseArray");
    assert(base_array.size() == base_array_size && "Array base_array incorrect for GenerateBaseArray");
    assert(base_array[0].size() == element_size && "Array base_array[i] incorrect for GenerateBaseArray");
    assert(base_r.size() == base_array_size && "Array base_r incorrect for GenerateBaseArray");

    // generate random r
    vector<myType> base_r_next, base_r_prev;
    vector<myType> tempA(element_size), tempAPrev(element_size), tempOffset(element_size);
    size_t offset;
    for (int i = 0; i < base_array_size; ++i)
    {
        base_r_next.push_back(aes_next->get64Bits() % base_array_size);
        base_r_prev.push_back(aes_prev->get64Bits() % base_array_size);

        base_r[i].first = base_r_next[i];
        base_r[i].second = base_r_prev[i];

        // rotate a
        for (size_t j = 0; j < element_size; ++j)
        {
            offset = (j + base_r_next[i] + base_r_prev[i]) % element_size;
            tempA[offset] = a[j].first + aes_prev->get64Bits() - aes_next->get64Bits();
        }

        // send tempA to previous party and receive tempAPrev
        tempAPrev.clear();
        tempAPrev.resize(element_size);
        thread *threads = new thread[2];
        threads[0] = thread(sendVector<myType>, ref(tempA), nextParty(partyNum), element_size);
        threads[1] = thread(receiveVector<myType>, ref(tempAPrev), prevParty(partyNum), element_size);
        for (int i = 0; i < 2; i++)
            threads[i].join();
        delete[] threads;

        // generate base_array
        for (size_t j = 0; j < element_size; ++j)
        {
            size_t offset = (j + base_r_next[i]) % element_size;
            tempOffset[offset] = tempAPrev[j];
        }

        base_array[i].assign(tempOffset.begin(), tempOffset.end());
    }
}

void funcBatchArrayAccess(const vector<vector<myType>> &base_array, const RSSVectorMyType &base_r, const RSSVectorMyType &index, RSSVectorMyType &c, size_t base_array_size, size_t index_size, size_t element_size)
{
    log_print("funcBatchArrayAccess");
    assert(base_array.size() == base_array_size && "Array base_array incorrect for BatchArrayAccess");
    assert(base_array[0].size() == element_size && "Array base_array[i] incorrect for BatchArrayAccess");
    assert(base_r.size() == base_array_size && "Array base_r incorrect for BatchArrayAccess");
    assert(index.size() == index_size && "Array index incorrect for BatchArrayAccess");
    assert(c.size() == index_size && "Array c incorrect for BatchArrayAccess");

    // offset index
    RSSVectorMyType tempIndex(index_size);
    vector<myType> tempIndexVector, tempIndexPrevVector(index_size);
    for (int i = 0; i < index_size; ++i)
    {
        size_t j = i % base_array_size;
        // first + r_prev
        tempIndex[i].first = index[i].first + base_r[j].second;
        // second + r_next
        tempIndex[i].second = index[i].second + base_r[j].first;

        tempIndexVector.push_back(tempIndex[i].first);
    }

    // send tempIndex.first to previous party and receive tempIndexPrev
    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(tempIndexVector), nextParty(partyNum), index_size);
    threads[1] = thread(receiveVector<myType>, ref(tempIndexPrevVector), prevParty(partyNum), index_size);

    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    // reconstruct h
    vector<size_t> h(index_size);
    for (int i = 0; i < index_size; ++i)
    {
        myType hMyType = tempIndexPrevVector[i] + tempIndex[i].first + tempIndex[i].second;
        h[i] = hMyType % element_size;
        if (h[i] > element_size - 1)
            error("index is out of range!");
    }

    // compute c
    vector<myType> a_h(index_size), b(index_size), tempA(element_size);
    myType t;
    for (int i = 0; i < index_size; ++i)
    {
        tempA.resize(element_size);
        tempA.assign(base_array[i % base_array_size].begin(), base_array[i % base_array_size].end());

        t = aes_prev->get64Bits() - aes_next->get64Bits();
        a_h[i] = tempA[h[i]] + t;
    }

    // send a_h to prev party and receive c
    thread *threads2 = new thread[2];
    threads2[0] = thread(sendVector<myType>, ref(a_h), prevParty(partyNum), index_size);
    threads2[1] = thread(receiveVector<myType>, ref(b), nextParty(partyNum), index_size);
    for (int i = 0; i < 2; i++)
        threads2[i].join();
    delete[] threads2;

    // set c.first = a[h], c.second = next party's a[h]
    for (int i = 0; i < index_size; ++i)
    {
        c[i].first = a_h[i];
        c[i].second = b[i];
    }
}

void funcBatchArrayAccess(const vector<myType> &base_array, const RSSMyType &base_r, const RSSMyType &index, RSSMyType &c, size_t element_size)
{
    log_print("funcBatchArrayAccess");
    assert(base_array.size() == element_size && "Array base_array incorrect for BatchArrayAccess");

    // offset index
    // offset index
    RSSMyType tempIndex;
    tempIndex.first = index.first + base_r.second;
    tempIndex.second = index.second + base_r.first;
    // send tempIndex.first to previous party and receive tempIndexPrev
    vector<myType> tempIndexVector(1, tempIndex.first);
    vector<myType> tempIndexPrevVector(1);

    // send tempIndex.first to previous party and receive tempIndexPrev
    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(tempIndexVector), nextParty(partyNum), 1);
    threads[1] = thread(receiveVector<myType>, ref(tempIndexPrevVector), prevParty(partyNum), 1);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    // reconstruct h
    size_t h = (tempIndexPrevVector[0] + tempIndex.first + tempIndex.second) % element_size;
    if (h > element_size - 1)
        error("index is out of range!");

    // compute c
    myType t, a_h, b;
    t = aes_prev->get64Bits() - aes_next->get64Bits();
    a_h = base_array[h] + t;

    // send a_h to prev party
    vector<myType> a_h_vector(1, a_h), b_vector(1);

    // send a_h to prev party and receive c
    thread *threads2 = new thread[2];
    threads2[0] = thread(sendVector<myType>, ref(a_h_vector), prevParty(partyNum), 1);
    threads2[1] = thread(receiveVector<myType>, ref(b_vector), nextParty(partyNum), 1);
    for (int i = 0; i < 2; i++)
        threads2[i].join();
    delete[] threads2;

    // set c.first = a[h], c.second = next party's a[h]
    c.first = a_h;
    c.second = b_vector[0];
}

void funcGenerateBaseArrayVector(const vector<RSSVectorMyType> &a, vector<vector<myType>> &base_array, RSSVectorMyType &base_r, size_t array_size, size_t base_array_size, size_t element_size)
{
    log_print("funcGenerateBaseArrayVector");
    assert(a.size() == array_size && "Array a incorrect for GenerateBaseArray");
    assert(base_array.size() == base_array_size && "Array base_array incorrect for GenerateBaseArray");
    assert(base_array[0].size() == array_size * element_size && "Array base_array incorrect for GenerateBaseArray");
    assert(base_r.size() == base_array_size && "Array base_r incorrect for GenerateBaseArray");

    // generate random r
    vector<myType> base_r_next, base_r_prev;
    vector<myType> tempA(array_size * element_size), tempAPrev(array_size * element_size);
    vector<myType> tempOffset(array_size * element_size);
    size_t offset;
    for (size_t i = 0; i < base_array_size; ++i)
    {
        base_r_next.push_back(aes_next->get64Bits() % array_size);
        base_r_prev.push_back(aes_prev->get64Bits() % array_size);

        base_r[i].first = base_r_next[i];
        base_r[i].second = base_r_prev[i];

        // rotate a
        for (size_t j = 0; j < array_size; ++j)
        {
            offset = (j + base_r_next[i] + base_r_prev[i]) % array_size;
            for (size_t k = 0; k < element_size; ++k)
                tempA[offset * element_size + k] = a[j][k].first + aes_prev->get64Bits() - aes_next->get64Bits();
        }

        // send tempA to previous party and receive tempAPrev
        tempAPrev.clear();
        tempAPrev.resize(array_size * element_size);
        thread *threads = new thread[2];
        threads[0] = thread(sendVector<myType>, ref(tempA), nextParty(partyNum), array_size * element_size);
        threads[1] = thread(receiveVector<myType>, ref(tempAPrev), prevParty(partyNum), array_size * element_size);
        for (int i = 0; i < 2; i++)
            threads[i].join();
        delete[] threads;

        // generate base_array
        for (size_t j = 0; j < array_size; ++j)
        {
            offset = (j + base_r_next[i]) % array_size;
            for (size_t k = 0; k < element_size; ++k)
                tempOffset[offset * element_size + k] = tempAPrev[j * element_size + k];
        }

        base_array[i].assign(tempOffset.begin(), tempOffset.end());
    }
}

void funcBatchArrayAccessVector(const vector<vector<myType>> &base_array, const RSSVectorMyType &base_r, const RSSVectorMyType &index, vector<RSSVectorMyType> &c, size_t array_size, size_t base_array_size, size_t index_size, size_t element_size)
{
    log_print("funcBatchArrayAccessVector");
    assert(base_array.size() == base_array_size && "Array base_array incorrect for BatchArrayAccess");
    assert(base_array[0].size() == array_size * element_size && "Array base_array incorrect for BatchArrayAccess");
    assert(base_r.size() == base_array_size && "Array base_r incorrect for BatchArrayAccess");
    assert(index.size() == index_size && "Array index incorrect for BatchArrayAccess");
    assert(c.size() == index_size && "Array c incorrect for BatchArrayAccess");

    // offset index
    RSSVectorMyType tempIndex(index_size);
    vector<myType> tempIndexVector, tempIndexPrevVector(index_size);
    for (int i = 0; i < index_size; ++i)
    {
        size_t j = i % base_array_size;
        // first + r_prev
        tempIndex[i].first = index[i].first + base_r[j].second;
        // second + r_next
        tempIndex[i].second = index[i].second + base_r[j].first;

        tempIndexVector.push_back(tempIndex[i].first);
    }

    // send tempIndex.first to previous party and receive tempIndexPrev
    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(tempIndexVector), nextParty(partyNum), index_size);
    threads[1] = thread(receiveVector<myType>, ref(tempIndexPrevVector), prevParty(partyNum), index_size);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    // reconstruct h
    vector<size_t> h(index_size);
    for (int i = 0; i < index_size; ++i)
    {
        myType hMyType = tempIndexPrevVector[i] + tempIndex[i].first + tempIndex[i].second;
        h[i] = hMyType % array_size;

        if (h[i] > array_size - 1)
            error("index is out of range!");
    }

    // compute c
    vector<myType> a_h_vector(index_size * element_size), b_vector(index_size * element_size);
    myType t;
    for (int i = 0; i < index_size; ++i)
    {
        vector<myType> tempA(array_size * element_size);
        tempA.assign(base_array[i % base_array_size].begin(), base_array[i % base_array_size].end());

        for (size_t j = 0; j < element_size; ++j)
        {
            // randomize the value of base_array[h[i]][j], local generate t, where t_0 + t_1 + t_2 = 0
            t = aes_prev->get64Bits() - aes_next->get64Bits();
            a_h_vector[i * element_size + j] = tempA[h[i] * element_size + j] + t;
        }
    }

    // send a_h to prev party and receive c
    thread *threads2 = new thread[2];
    threads2[0] = thread(sendVector<myType>, ref(a_h_vector), prevParty(partyNum), index_size * element_size);
    threads2[1] = thread(receiveVector<myType>, ref(b_vector), nextParty(partyNum), index_size * element_size);
    for (int i = 0; i < 2; i++)
        threads2[i].join();
    delete[] threads2;

    // set c.first = a[h], c.second = next party's a[h]
    for (int i = 0; i < index_size; ++i)
    {
        for (size_t j = 0; j < element_size; ++j)
        {
            c[i][j].first = a_h_vector[i * element_size + j];
            c[i][j].second = b_vector[i * element_size + j];
        }
    }
}

// Functions for batch array access, base_array from file
void funcBatchArrayAccessVector(const vector<myType> &base_array, const RSSMyType &base_r, const RSSMyType &index, RSSVectorMyType &c, size_t array_size, size_t element_size)
{
    log_print("funcBatchArrayAccessVector");
    assert(base_array.size() == array_size * element_size && "Array base_array incorrect for BatchArrayAccess");
    assert(c.size() == element_size && "Array c incorrect for BatchArrayAccess");

    // offset index
    RSSMyType tempIndex;
    tempIndex.first = index.first + base_r.second;
    tempIndex.second = index.second + base_r.first;
    // send tempIndex.first to previous party and receive tempIndexPrev
    vector<myType> tempIndexVector(1, tempIndex.first);
    vector<myType> tempIndexPrevVector(1);

    // send tempIndex.first to previous party and receive tempIndexPrev
    thread *threads = new thread[2];
    threads[0] = thread(sendVector<myType>, ref(tempIndexVector), nextParty(partyNum), 1);
    threads[1] = thread(receiveVector<myType>, ref(tempIndexPrevVector), prevParty(partyNum), 1);
    for (int i = 0; i < 2; i++)
        threads[i].join();
    delete[] threads;

    // reconstruct h
    size_t h = (tempIndexPrevVector[0] + tempIndex.first + tempIndex.second) % array_size;
    // get a[h]
    if (h > array_size - 1)
        error("index is out of range!");

    // compute c
    vector<myType> a_h_vector(element_size), b_vector(element_size);
    myType t;

    for (size_t i = 0; i < element_size; ++i)
    {
        // randomize the value of base_array[h[i]][j], local generate t, where t_0 + t_1 + t_2 = 0
        t = aes_prev->get64Bits() - aes_next->get64Bits();
        a_h_vector[i] = base_array[h * element_size + i] + t;
    }

    // send a_h to prev party and receive c
    thread *threads2 = new thread[2];
    threads2[0] = thread(sendVector<myType>, ref(a_h_vector), prevParty(partyNum), element_size);
    threads2[1] = thread(receiveVector<myType>, ref(b_vector), nextParty(partyNum), element_size);
    for (int i = 0; i < 2; i++)
        threads2[i].join();
    delete[] threads2;

    // set c.first = a[h], c.second = next party's a[h]
    for (size_t i = 0; i < element_size; ++i)
    {
        c[i].first = a_h_vector[i];
        c[i].second = b_vector[i];
    }
}

//=========================================== DEBUG ======================================================
void debugTrunc()
{
    vector<myType> data = {floatToMyType(4), floatToMyType(2.5), floatToMyType(0.5547), floatToMyType(-4)};

    RSSVectorMyType a(data.size());
    funcGetShares(a, data);
    print_vector(a, "SIGNED", "Trunc Original (SIGNED):", data.size());
    // funcTruncatePublic(a, 2.5, data.size());
    funcTruncate(a, 2, data.size());
    print_vector(a, "SIGNED", "Trunc Result (SIGNED): ", data.size());
    print_vector(a, "FLOAT", "Trunc Result (FLOAT): ", data.size());
}

void debugMatMul()
{
    size_t rows = 2, columns = 2, common_dim = 3;
    vector<myType> data_a = {floatToMyType(1), floatToMyType(1),
                             floatToMyType(4), floatToMyType(2),
                             floatToMyType(5), floatToMyType(3)};
    vector<myType> data_b = {floatToMyType(3), floatToMyType(1),
                             floatToMyType(4), floatToMyType(5),
                             floatToMyType(2), floatToMyType(6)};

    size_t size = rows * columns;
    RSSVectorMyType a(rows * common_dim), b(common_dim * columns), c(size);
    funcGetShares(a, data_a);
    funcGetShares(b, data_b);
    funcMatMul(a, b, c, rows, common_dim, columns, 1, 1, FLOAT_PRECISION);
    print_vector(c, "FLOAT", "MatMul Result: ", size);
}

void debugDotProd()
{
    int size = 12;
    vector<myType> data_a = {floatToMyType(3), floatToMyType(-4), floatToMyType(5), floatToMyType(6),
                             floatToMyType(10), floatToMyType(20), floatToMyType(30), floatToMyType(40),
                             floatToMyType(50), floatToMyType(60), floatToMyType(70), floatToMyType(80)};
    vector<myType> data_b = {floatToMyType(1), floatToMyType(2), floatToMyType(3), floatToMyType(4),
                             floatToMyType(5), floatToMyType(6), floatToMyType(7), floatToMyType(8),
                             floatToMyType(9), floatToMyType(10), floatToMyType(11), floatToMyType(12)};
    // size_t size = 100;
    // vector<myType> data_a(size), data_b(size);
    // cout << "start generating data" << endl;
    // for (int i = 0; i < size; ++i)
    // {
    //     data_a[i] = floatToMyType(getrandom(1, 1 + i % 10));
    //     data_b[i] = floatToMyType(getrandom(1, 1 + i % 10));
    // }
    // cout << "end generating data" << endl;

    RSSVectorMyType a(size), b(size), c(size);
    funcGetShares(a, data_a);
    funcGetShares(b, data_b);
    funcDotProduct(a, b, c, size, true, FLOAT_PRECISION);
    print_vector(c, "FLOAT", "DotProd Result: ", size);
}

void debugPC()
{
    vector<myType> plain_m{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<myType> plain_r{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    vector<smallType> plain_beta{1, 0, 1, 0, 0, 0, 1, 1, 0, 1};
    size_t size = plain_m.size();
    size_t sizeLong = size * BIT_SIZE;
    assert(plain_r.size() == plain_m.size() && "Error in debugPC");

    RSSVectorSmallType beta(size), shares_m(sizeLong);
    vector<smallType> reconst_betaP(size), betaPrime(size);
    funcGetShares(beta, plain_beta);

    vector<smallType> bits_of_m(sizeLong);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < BIT_SIZE; ++j)
            bits_of_m[i * BIT_SIZE + j] = (smallType)((plain_m[i] >> (BIT_SIZE - 1 - j)) & 1);

    funcGetShares(shares_m, bits_of_m);
    funcPrivateCompare(shares_m, plain_r, beta, betaPrime, size);

#if (!LOG_DEBUG)
    cout << "BetaPrime: \t ";
    for (int i = 0; i < size; ++i)
        cout << (int)betaPrime[i] << " ";
    cout << endl;
    cout << "Beta: \t\t";
    for (int i = 0; i < size; ++i)
        cout << (int)plain_beta[i] << " ";
    cout << endl;
    cout << "m: \t\t";
    for (int i = 0; i < size; ++i)
        cout << (int)plain_m[i] << " ";
    cout << endl;
    cout << "r: \t\t";
    for (int i = 0; i < size; ++i)
        cout << (int)plain_r[i] << " ";
    cout << endl;
    cout << "m-r: \t\t";
    for (int i = 0; i < size; ++i)
        cout << (int)plain_m[i] - (int)plain_r[i] << " ";
    cout << endl;
#endif
}

void debugWrap()
{
    size_t size = 5;
    RSSVectorMyType a(size);
    RSSVectorSmallType theta(size);
    vector<smallType> reconst(size);

    myType interesting = MINUS_ONE / 3;
    a[0] = make_pair(0, 0);
    a[1] = make_pair(interesting, interesting);
    interesting += 1;
    a[2] = make_pair(interesting, interesting);
    interesting = ((MINUS_ONE / 3) << 1);
    a[3] = make_pair(interesting, interesting);
    interesting += 1;
    a[4] = make_pair(interesting, interesting);

    funcWrap(a, theta, size);

#if (!LOG_DEBUG)
    cout << "a: \t\t ";
    for (int i = 0; i < size; ++i)
        cout << (int)3 * a[i].first << "(" << (int)a[i].first << ", " << (int)a[i].second << ") ";
    cout << endl;
    funcReconstruct(theta, reconst, size, "Theta", true);
#endif
}

void debugReLUPrime()
{
    vector<myType> data_a = {floatToMyType(9), floatToMyType(-12), floatToMyType(0), floatToMyType(-2), floatToMyType(3)};
    size_t size = data_a.size();
    RSSVectorMyType a(size);
    RSSVectorSmallType b(size);
    vector<myType> reconst_a(size);
    vector<smallType> reconst_b(size);

    funcGetShares(a, data_a);
    funcRELUPrime(a, b, size);

#if (!LOG_DEBUG)
    funcReconstruct(a, reconst_a, size, "a", true);
    funcReconstructBit(b, reconst_b, size, "b", true);
#endif
}

void debugReLU()
{
    vector<myType> data_a = {0, 1, 2, 3, 4, 5, 6, 7};
    size_t size = data_a.size();
    RSSVectorMyType a(size), b(size);
    RSSVectorSmallType aPrime(size);
    vector<myType> reconst_b(size);
    vector<smallType> reconst_b_prime(size);

    funcGetShares(a, data_a);

    for (int i = size / 2; i < size; ++i)
    {
        a[i].first = a[i].first << BIT_SIZE - 3;
        a[i].second = a[i].second << BIT_SIZE - 3;
    }
    print_vector(a, "FLOAT", "a float: ", size);

    funcRELU(a, aPrime, b, size);

#if (!LOG_DEBUG)
    funcReconstruct(a, data_a, size, "a", true);
    funcReconstruct(b, reconst_b, size, "ReLU", true);
    funcReconstructBit(aPrime, reconst_b_prime, size, "ReLUPrime", true);
#endif
}

void debugDivision()
{
    // 若输出结果不正确，（1）尝试修改myType类型为uint64_t，（2）降低PRECISE_DIVISION的值
    vector<myType> data_a = {floatToMyType(3.2), floatToMyType(2), floatToMyType(1), floatToMyType(200)},
                   data_b = {floatToMyType(9.7), floatToMyType(0.5), floatToMyType(2), floatToMyType(10)};
    size_t size = data_a.size();
    RSSVectorMyType a(size), b(size), quotient(size);
    vector<myType> reconst(size);

    funcGetShares(a, data_a);
    funcGetShares(b, data_b);
    funcDivision(a, b, quotient, size);

#if (!LOG_DEBUG)
    funcReconstruct(a, reconst, size, "a", true);
    funcReconstruct(b, reconst, size, "b", true);
    // funcReconstruct(quotient, reconst, size, "Quot", true);
    // print_myType(reconst[0], "Quotient[0]", "FLOAT");
    print_vector(quotient, "FLOAT", "Quotient", size);
#endif
}

void debugSelectBitShares()
{
    size_t rows = 4;
    size_t columns = 3;
    vector<smallType> a0 = {1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0};
    vector<smallType> a1 = {0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                            1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
                            1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1};
    vector<smallType> rp = {0, 0, 1, 1};
    RSSVectorSmallType x(rows * columns), y(rows * columns * columns), z(rows), answer(rows * columns);
    funcGetShares(x, a0);
    funcGetShares(y, a1);
    funcGetShares(z, rp);

#if (!LOG_DEBUG)
    funcReconstructBit(x, a1, x.size(), "x", true);
    funcReconstructBit(y, a1, y.size(), "y", true);
    funcReconstructBit(z, a1, z.size(), "z", true);
#endif

    for (int i = 0; i < 3; ++i)
    {
        funcSelectBitShares(x, y, z, answer, rows, columns, i);
#if (!LOG_DEBUG)
        funcReconstructBit(answer, a1, answer.size(), "a", true);
#endif
    }
}

void debugSelectShares()
{

    vector<smallType> bits = {1, 0, 0, 1, 1, 1, 0, 1, 1, 0};
    size_t size = bits.size();
    vector<myType> data = {1, 29, 10, 2938, 27, -1, -23, 12, 2, 571}, reconst(size);
    assert(size == data.size() && "Size mismatch");
    RSSVectorMyType a(size), selection(size);
    RSSVectorSmallType b(size);

    funcGetShares(a, data);
    funcGetShares(b, bits);

    funcSelectShares(a, b, selection, size);

#if (!LOG_DEBUG)
    funcReconstruct(a, reconst, size, "a", true);
    funcReconstructBit(b, bits, size, "b", true);
    funcReconstruct(selection, reconst, size, "Sel'd", true);
#endif
}

void debugMaxpool()
{
    size_t rows = 5;
    size_t columns = 3;
    size_t size = rows * columns;
    vector<myType> data = {1, 2, 3,
                           3, 1, 2,
                           1, 5, 3,
                           5, 1, 6,
                           6, 3, 9},
                   reconst(size);
    RSSVectorMyType a(size), max(rows);
    RSSVectorSmallType maxPrime(rows * columns);
    vector<smallType> reconst_maxPrime(maxPrime.size());
    funcGetShares(a, data);
    funcMaxpool(a, max, maxPrime, rows, columns);

#if (!LOG_DEBUG)
    funcReconstruct(a, reconst, size, "a", true);
    funcReconstruct(max, reconst, rows, "val", true);
    funcReconstructBit(maxPrime, reconst_maxPrime, rows * columns, "maxP", true);
#endif
}

void debugPow()
{
    // for 2^x, x = 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, ..., 63, 0
    vector<myType> data = {4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456,
                           // x = 30, 32 , 34, 36, 38, 40, 42, 44, 46, 48
                           1073741824, 4294967296, 17179869184, 68719476736, 274877906944, 1099511627776, 4398046511104, 17592186044416, 70368744177664, 281474976710656,
                           // x = 50, 52, 54, 56, 58, 60, 62, 63
                           1125899906842624, 4503599627370496, 18014398509481984, 72057594037927936, 288230376151711744, 1152921504606846976, 4611686018427387904, 9223372036854775808,
                           // 2^x = 2^63 - 1
                           18446744073709551615};

    size_t size = data.size();
    RSSVectorMyType a(size);
    vector<smallType> alpha(size);
    vector<myType> reconst(size);

    funcGetShares(a, data);
    funcPow(a, alpha, size);

#if (!LOG_DEBUG)
    funcReconstruct(a, reconst, size, "a", true);
    cout << "Alpha: \t\t";
    for (int i = 0; i < size; ++i)
        cout << (int)alpha[i] << " ";
    cout << endl;
#endif
}

void debugInvSqrt()
{
    size_t size = 3;
    vector<myType> data_a = {floatToMyType(0.3333), floatToMyType(112), floatToMyType(11)};

    RSSVectorMyType a(size), b(size), b_approx(size);
    funcGetShares(a, data_a);
    // funcInvSqrt(a, b, size);
    cout << "InvSqrt Time: \t" << funcTime(funcInvSqrt, a, b, size) << endl;

#if (!LOG_DEBUG)
    print_vector(a, "FLOAT", "Original: ", size);
    cout << "Plaintext InvSqrt: " << 1.0 / sqrt(0.333) << " " << 1.0 / sqrt(112) << " " << 1.0 / sqrt(11) << endl;
    print_vector(b, "FLOAT", "InvSqrt Result: ", size);
#endif
}

void debugInv()
{
    size_t size = 3;
    vector<myType> data_a = {floatToMyType(0.3333), floatToMyType(0.2), floatToMyType(11)};

    RSSVectorMyType a(size), b(size);
    funcGetShares(a, data_a);
    funcInv(a, b, size);

#if (!LOG_DEBUG)
    print_vector(a, "FLOAT", "Original: ", size);
    cout << "Plaintext Inv: " << 1.0 / 0.3333 << " " << 1.0 / 0.2 << " " << 1.0 / 11 << endl;
    print_vector(b, "FLOAT", "Inv Result: ", size);
#endif
}

void debugInnerProd()
{
    size_t size = 4;
    vector<myType> data_a = {floatToMyType(3), floatToMyType(-4), floatToMyType(5), floatToMyType(6)};
    vector<myType> data_b = {floatToMyType(1), floatToMyType(2), floatToMyType(3), floatToMyType(4)};

    RSSVectorMyType a(size), b(size);
    funcGetShares(a, data_a);
    funcGetShares(b, data_b);
    RSSMyType c;
    funcInnerProduct(a, b, c, size, FLOAT_PRECISION);
    RSSVectorMyType c_temp(1, c);

#if (!LOG_DEBUG)
    print_vector(c_temp, "FLOAT", "DotProd Result: ", 1);
#endif
}

void debugSoftmax()
{

    vector<myType> data_a = {floatToMyType(2), floatToMyType(3), floatToMyType(5),
                             floatToMyType(4), floatToMyType(5), floatToMyType(6)};
    size_t size = data_a.size();

    RSSVectorMyType a(size), b(size);
    funcGetShares(a, data_a);
    // funcSoftmax(a, b, size);
    cout << "Softmax Time: \t" << funcTime(funcSoftmax, a, b, size) << endl;

#if (!LOG_DEBUG)
    vector<double> data_b = {2, 3, 5, 4, 5, 6};
    double exp_res = 0;
    for (int i = 0; i < data_b.size(); ++i)
        exp_res += exp(data_b[i]);

    cout << "Plaintext: \t";
    for (int i = 0; i < size; ++i)
        cout << exp(data_b[i]) / exp_res << " ";
    cout << endl;
    print_vector(a, "FLOAT", "Original: ", size);
    print_vector(b, "FLOAT", "Softmax Result: ", size);
#endif
}

void debugArrayAccess()
{

    vector<myType> data_a = {1, 2, 3, 4, 5, 6};
    size_t size = data_a.size();
    myType data_index = 3;

    RSSVectorMyType a(size);
    RSSVectorMyType index(1);
    funcGetShares(a, data_a);
    funcGetShares(index, {data_index});
    RSSMyType c = std::make_pair(0, 0);
    funcArrayAccess(a, index[0], c, size);
    RSSVectorMyType c_temp(1);
    c_temp[0] = c;

#if (!LOG_DEBUG)
    print_myType(data_index, "Index1", "SIGNED");
    print_vector(a, "SIGNED", "ArrayAccess Original: ", size);
    print_vector(c_temp, "SIGNED", "ArrayAccess Result: ", 1);
#endif
}

void debugArrayAccessVector()
{
    size_t size = 4;

    vector<myType> data_1 = {floatToMyType(1.3), floatToMyType(-2.4), floatToMyType(1.5)};
    vector<myType> data_2 = {floatToMyType(3.6), floatToMyType(0.1), floatToMyType(2.2)};
    vector<myType> data_3 = {floatToMyType(3), floatToMyType(4), floatToMyType(5)};
    vector<myType> data_4 = {floatToMyType(6), floatToMyType(7), floatToMyType(8)};

    size_t element_size = data_1.size();
    myType data_index = 3;
    print_myType(data_index, "Index", "SIGNED");

    RSSVectorMyType a_1(element_size), a_2(element_size), a_3(element_size), a_4(element_size);
    RSSVectorMyType index(1);
    funcGetShares(a_1, data_1);
    print_vector(a_1, "FLOAT", "a_1: ", element_size);
    funcGetShares(a_2, data_2);
    print_vector(a_2, "FLOAT", "a_2: ", element_size);
    funcGetShares(a_3, data_3);
    print_vector(a_3, "FLOAT", "a_3: ", element_size);
    funcGetShares(a_4, data_4);
    print_vector(a_4, "FLOAT", "a_4: ", element_size);
    funcGetShares(index, {data_index});

    RSSVectorMyType c(element_size);
    funcArrayAccessVector({a_1, a_2, a_3, a_4}, index[0], c, size, element_size);
    print_vector(c, "FLOAT", "ArrayAccessVector Result: ", element_size);
}

void debugBN()
{
    // Values over and above 10, use lower fixed-point precision. 13 bits + another 2-3 powers of 2 is too much.
    vector<myType> data_a = {floatToMyType(3.2), floatToMyType(9.8), floatToMyType(6.4), floatToMyType(9.8)}, data_b = {floatToMyType(9.8), floatToMyType(3.2)};
    size_t size = data_a.size();
    RSSVectorMyType a(size), b(data_b.size()), quotient(size);
    vector<myType> reconst(size);

    funcGetShares(a, data_a);
    funcGetShares(b, data_b);
    funcBatchNorm(a, b, quotient, 2, 2);

#if (!LOG_DEBUG)
    funcReconstruct(a, reconst, size, "a", true);
    funcReconstruct(b, reconst, data_b.size(), "b", true);
    funcReconstruct(quotient, reconst, size, "Quot", true);
    print_vector(quotient, "FLOAT", "Quotient", size);
#endif
}

void debugForbeniusNorm()
{
    vector<myType> data_a = {floatToMyType(3), floatToMyType(-4), floatToMyType(5), floatToMyType(6)};
    size_t size = data_a.size();
    RSSVectorMyType a(size), b(size);
    vector<myType> reconst(size);

    funcGetShares(a, data_a);
    funcForbeniusNorm(a, b, size);

#if (!LOG_DEBUG)
    funcReconstruct(a, reconst, size, "a", true);
    double result = sqrt(3 * 3 + 4 * 4 + 5 * 5 + 6 * 6);
    cout << "Plaintext ForbeniusNorm: " << 3 / result << " " << -4 / result << " " << 5 / result << " " << 6 / result << endl;
    print_vector(b, "FLOAT", "ForbeniusNorm Result: ", size);
#endif
}

void debugBatchArrayAccess()
{
    size_t base_array_size = 3, element_size = 5;
    vector<vector<myType>> base_array(base_array_size, vector<myType>(element_size, 0));
    RSSVectorMyType base_r(base_array_size);
    vector<myType> data_index = {1, 3, 0};
    RSSVectorMyType index(data_index.size());
    RSSVectorMyType c(data_index.size());

    vector<myType> data = {floatToMyType(1.3), floatToMyType(-2.4), floatToMyType(1.5), floatToMyType(3.6), floatToMyType(2)};

    RSSVectorMyType a(element_size);
    funcGetShares(a, data);
    funcGetShares(index, data_index);

    funcGenerateBaseArray(a, base_array, base_r, base_array_size, element_size);
    funcBatchArrayAccess(base_array, base_r, index, c, base_array_size, data_index.size(), element_size);

    cout << "Original Data: " << endl;
    print_vector(a, "FLOAT", "a: ", element_size);
    cout << "Index: " << endl;
    for (int i = 0; i < data_index.size(); ++i)
        cout << data_index[i] << " ";
    cout << endl;

    print_vector(c, "FLOAT", "BatchArrayAccess Result: ", data_index.size());

    RSSMyType d = std::make_pair(0, 0);
    funcBatchArrayAccess(base_array[0], base_r[0], index[2], d, element_size);
    RSSVectorMyType d_temp(1);
    d_temp[0] = d;
    print_vector(d_temp, "FLOAT", "BatchArrayAccess2 Result: ", 1);
}

void debugBatchArrayAccessVector()
{
    size_t array_size = 4, base_array_size = 3, element_size = 3;
    vector<vector<myType>> base_array(base_array_size, vector<myType>(array_size * element_size, 0));
    RSSVectorMyType base_r(base_array_size);
    vector<myType> data_index = {1, 3, 0};
    RSSVectorMyType index(data_index.size());
    vector<RSSVectorMyType> c(data_index.size(), RSSVectorMyType(element_size));

    vector<myType> data_1 = {floatToMyType(1.3), floatToMyType(-2.4), floatToMyType(1.5)};
    vector<myType> data_2 = {floatToMyType(3.6), floatToMyType(0.1), floatToMyType(2.2)};
    vector<myType> data_3 = {floatToMyType(3), floatToMyType(4), floatToMyType(5)};
    vector<myType> data_4 = {floatToMyType(6), floatToMyType(7), floatToMyType(8)};

    RSSVectorMyType a_1(element_size), a_2(element_size), a_3(element_size), a_4(element_size);
    funcGetShares(a_1, data_1);
    funcGetShares(a_2, data_2);
    funcGetShares(a_3, data_3);
    funcGetShares(a_4, data_4);
    funcGetShares(index, data_index);

    funcGenerateBaseArrayVector({a_1, a_2, a_3, a_4}, base_array, base_r, array_size, base_array_size, element_size);
    funcBatchArrayAccessVector(base_array, base_r, index, c, array_size, base_array_size, data_index.size(), element_size);

    cout << "Original Data: " << endl;
    print_vector(a_1, "FLOAT", "a_1: ", element_size);
    print_vector(a_2, "FLOAT", "a_2: ", element_size);
    print_vector(a_3, "FLOAT", "a_3: ", element_size);
    print_vector(a_4, "FLOAT", "a_4: ", element_size);

    for (int i = 0; i < data_index.size(); ++i)
    {
        cout << "Index: " << data_index[i] << endl;
        print_vector(c[i], "FLOAT", "BatchArrayAccessVector Result: ", element_size);
    }

    RSSVectorMyType d(element_size);
    funcBatchArrayAccessVector(base_array[0], base_r[0], index[0], d, array_size, element_size);
    print_vector(d, "FLOAT", "BatchArrayAccessVector2 Result: ", element_size);
}

void testRSS()
{
    vector<myType> data = {781, 562, 15, 9962, 2 ^ 10};
    RSSVectorMyType a(data.size());
    funcGetShares(a, data);

    log_print("Get Shares");
    for (int i = 0; i < data.size(); ++i)
    {
        cout << "a[" << i << "].first = " << a[i].first << endl;
        cout << "a[" << i << "].second = " << a[i].second << endl;
    }

    vector<myType> data_reconst(data.size());
    funcReconstruct(a, data_reconst, data.size(), "Reconst RSSShares Result (myType)", true);
    for (int i = 0; i < data.size(); ++i)
        cout << "data_reconst[" << i << "] = " << data_reconst[i] << endl;
    print_vector(a, "FLOAT", "Reconst RSSShares Result (FLOAT)", data.size());
}