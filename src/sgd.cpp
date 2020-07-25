/**
 \file      sgd.cpp
 \author    Johannes Unruh johannes.unruh@fau.de
 \copyright Johannes Unruh
            This program is free software: you can redistribute it and/or modify
            it under the terms of the GNU Lesser General Public License as published
            by the Free Software Foundation, either version 3 of the License, or
            (at your option) any later version.
            ABY is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
            GNU Lesser General Public License for more details.
            You should have received a copy of the GNU Lesser General Public License
            along with this program. If not, see <http://www.gnu.org/licenses/>.
 \brief     Stochastic Gradient Descent using the ABY framework.
 */

#include "sgd.h"

#include <iostream>
#include <fstream>
#include <random>
#include <math.h>
#include "../extern/ABY/src/abycore/circuit/arithmeticcircuits.h"
#include "../extern/ABY/src/abycore/sharing/sharing.h"
#include "../extern/mnist/include/mnist/mnist_reader.hpp"
#include "../extern/mnist/include/mnist/mnist_utils.hpp"
#include "linear-model-generator.h"

#define INPUT_BITLEN  32

using namespace encsgd;

EncryptionEngine *
encryption_engine_new (e_role role, uint32_t precision, const std::string& address,
                       uint16_t port, seclvl seclvl, uint32_t nthreads, e_mt_gen_alg mt_alg)
{
    EncryptionEngine *engine = (EncryptionEngine*) malloc (sizeof(EncryptionEngine));

    uint32_t bitlen = INPUT_BITLEN;

    engine->party = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg);

    std::vector<Sharing*>& sharing = engine->party->GetSharings();

    engine->role = role;

    engine->precision = 1ull << precision;

    engine->shift = precision;

    engine->ac = (ArithmeticCircuit*) sharing[S_ARITH]->GetCircuitBuildRoutine();

    engine->bc = (BooleanCircuit*) sharing[S_BOOL]->GetCircuitBuildRoutine();

    engine->yc = (BooleanCircuit*) sharing[S_YAO]->GetCircuitBuildRoutine();

    return engine;
}

static void
get_plain_data (Matrix<double>& plain_X, Vector<double>& plain_y)
{
    std::ifstream features("../data/train.data");
    for (int i = 0; i < plain_X.rows(); i++)
        for (int j = 0; j < plain_X.cols(); j++)
            features >> plain_X(i,j);
    features.close();

    std::ifstream labels("../data/train.labels");
    for (int i = 0; i < plain_X.rows(); i++)
    {
        labels >> plain_y(i);
    }
    labels.close();
}

static void
generate_shared_data (EncryptionEngine *engine,
                      share ***s_X, share **s_y, share **s_w,
                      Matrix<double> plain_X, Vector<double> plain_y, Vector<double> plain_w)
{
    int rows = plain_X.rows();
    int columns = plain_X.cols();

    /* SETUP PHASE:
     * Generate the shares X_0 and X_1 of X, that is:
     * X = X_0 + X_1, with X_1 randomly generated
     * Same for w and y.
     */
    Matrix<uint32_t> X;
    Matrix<uint32_t> X_0;
    Matrix<uint32_t> X_1;
    Vector<uint32_t> y;
    Vector<uint32_t> y_0;
    Vector<uint32_t> y_1;
    Vector<uint32_t> w;
    Vector<uint32_t> w_0;
    Vector<uint32_t> w_1;

    plain_X *= engine->precision;
    X = plain_X.cast <uint32_t> ();
    X_1.setRandom (rows, columns);
    X_0 = X - X_1;

    plain_y *= engine->precision;
    y = plain_y.cast <uint32_t> ();
    y_1.setRandom (rows);
    y_0 = y - y_1;

    plain_w *= engine->precision;
    w = plain_w.cast <uint32_t> ();
    w_1.setRandom (columns);
    w_0 = w - w_1;

    if (engine->role == SERVER)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                s_X[i][j] = engine->ac->PutSharedINGate(X_0(i, j), INPUT_BITLEN);

        for (int i = 0; i < rows; i++)
            s_y[i] = engine->ac->PutSharedINGate(y_0(i), INPUT_BITLEN);

        for (int i = 0; i < columns; i++)
            s_w[i] = engine->ac->PutSharedINGate(w_0(i), INPUT_BITLEN);
    }
    else
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                s_X[i][j] = engine->ac->PutSharedINGate(X_1(i, j), INPUT_BITLEN);

        for (int i = 0; i < rows; i++)
            s_y[i] = engine->ac->PutSharedINGate(y_1(i), INPUT_BITLEN);

        for (int i = 0; i < columns; i++)
            s_w[i] = engine->ac->PutSharedINGate(w_1(i), INPUT_BITLEN);
    }
}

/* Returns z = ina*inb */
static share*
mul_q (EncryptionEngine *engine, share *ina, share *inb)
{
    share *ina_inv, *inb_inv, *ina_inv_yao, *inb_inv_yao, *ina_yao, *inb_yao, *mul_1, *mul_2, *ina_is_less_zero, *inb_is_less_zero, *res, *res_inv, *dec;
    share *threshold = engine->yc->PutCONSGate(1ull << (INPUT_BITLEN-1), INPUT_BITLEN);
    share *shift = engine->yc->PutCONSGate(engine->shift, INPUT_BITLEN);

    ina_inv = engine->ac->PutINVGate(ina);
    inb_inv = engine->ac->PutINVGate(inb);

    ina_yao = engine->yc->PutA2YGate(ina);
    inb_yao = engine->yc->PutA2YGate(inb);
    ina_inv_yao = engine->yc->PutA2YGate(ina_inv);
    inb_inv_yao = engine->yc->PutA2YGate(inb_inv);

    // if ina < 0 or inb < 0
    ina_is_less_zero = engine->yc->PutGTGate(ina_yao, threshold);
    inb_is_less_zero = engine->yc->PutGTGate(inb_yao, threshold);

    mul_1 = engine->yc->PutMUXGate(ina_inv_yao, ina_yao, ina_is_less_zero);
    mul_2 = engine->yc->PutMUXGate(inb_inv_yao, inb_yao, inb_is_less_zero);
    dec = engine->yc->PutXORGate (ina_is_less_zero, inb_is_less_zero);

    res = engine->yc->PutMULGate(mul_1, mul_2); //FIXME: Maybe Y2A, ArithmeticMul, A2Y is faster?
    res = engine->yc->PutBarrelRightShifterGate(res, shift);
    res_inv = engine->yc->PutINVGate(res); // FIXME: INV-Gate in yao circuit not precise? Maybe use arithmetic circuit instead?
    res = engine->yc->PutMUXGate(res_inv, res, dec);

    res = engine->ac->PutY2AGate(res, engine->bc);

    return res;
}

static share*
inner_prod(EncryptionEngine *engine, share **ina, share **inb, uint32_t columns)
{
    share **ret = (share**) malloc(sizeof(share*) * columns);

    // pairwise multiplication of all input values
    for (uint32_t i = 0; i < columns; i++)
    {
        ret[i] = mul_q(engine, ina[i], inb[i]);
    }

    // add up the individual multiplication results
    // in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
    for (uint32_t i = 1; i < columns; i++)
    {
        ret[0] = engine->ac->PutADDGate(ret[0], ret[i]);
    }

    return ret[0];
}

static share**
sgd (EncryptionEngine *engine, RegressionParams& params, share ***s_X,
     share **s_w, share **s_y, uint32_t *indices, int columns)
{
    share **temp = (share**) malloc (sizeof(share*) * columns);

    uint32_t learning_rate = (uint32_t) (params.learningRate * engine->precision);

    for (uint32_t k = 0; k < params.maxIterations; k++)
    {
        /* We setup the formula: [df(w)_i]l = x_i_l * (w*x - y)
         * i := indices[k]
         * In the following comments x:= x_i
         */

        // compute x*w
        temp[0] = inner_prod(engine, s_X[indices[k]], s_w, columns);
//        engine->ac->PutPrintValueGate(temp[0], "xw");

        // compute x*w-y
        temp[0] = engine->ac->PutSUBGate(temp[0], s_y[indices[k]]);
//        engine->ac->PutPrintValueGate(temp[0], "xw-y");

        // compute x_l(x*w-y)
        for (int i = 0; i < columns; i++)
        {
            temp[i] = mul_q (engine, temp[0], s_X[indices[k]][i]);
//            engine->ac->PutPrintValueGate(temp[i], "x_l(x*w-y)");
        }

        share *alpha = engine->ac->PutCONSGate(learning_rate, INPUT_BITLEN);

        // compute w = w - alpha * x_l(x*w-y)
        for (int i = 0; i < columns; i++)
        {
            temp[i] = mul_q (engine, alpha, temp[i]);
//            engine->ac->PutPrintValueGate(temp[i], "alpha*x_l(x*w-y)");
            s_w[i] = engine->ac->PutSUBGate(s_w[i], temp[i]);
//            engine->ac->PutPrintValueGate(s_w[i], "w=w-alpha*x_l(x*w-y)");
        }
    }

    free (temp);

    return s_w;
}

static double
empirical_risk (Matrix<double> X, Vector<double> y, Vector<double> w)
{
    Vector<double> error =  X * w - y;
    double l2 = error.norm();

    return l2;
}

static Vector<double>
plain_sgd (RegressionParams& params, Matrix<double> X, Vector<double> y, Vector<double> w, uint32_t *indices)
{
    Vector<double> grad;
    double error;
    double alpha = params.learningRate;

    for (uint32_t k = 0; k < params.maxIterations; k++)
    {
            error = X.row(indices[k]).dot(w);
            error -= y(indices[k]);
            grad = error * X.row(indices[k]);
            w = w - alpha * grad;

            if (k % 10 == 0)
                std::cout << "--- risk: " << empirical_risk (X, y, w) << "---\n";
    }

    return w;
}

int32_t main_sgd (e_role role, double learning_rate, uint32_t precision,
                  uint32_t max_iter, bool provide_data, const std::string& address,
                  uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
                  e_mt_gen_alg mt_alg)
{
    EncryptionEngine *engine = encryption_engine_new (role, precision, address, port, seclvl, nthreads,  mt_alg);

    int rows = 20;
    int columns = 3;

    Matrix<double> plain_X(rows, columns);
    Vector<double> plain_y(rows);
    Vector<double> plain_w(columns);

    if (provide_data)
        get_plain_data (plain_X, plain_y);
    else
    {   // Generate artificial data
        LinearModelGen gen;

        Vector<double> model(columns);
        for (int i = 0; i < columns; ++i)
        {
            model(i) = rand() * i % 10;
        }
        gen.setModel(model);

        gen.sample(plain_X, plain_y);
    }

    RegressionParams params;
    params.maxIterations = max_iter;
    params.learningRate = learning_rate;

    uint32_t indices[params.maxIterations];
    for (uint32_t i = 0; i < params.maxIterations; i++)
    {
        indices[i] = i % rows; // FIXME: indices[i] = rand() % rows; leads to silly results.
                               // Why? It seems rand() does interfere with
                               // something from ABY...
    }
    for (int i = 0; i < plain_w.size(); i++)
        plain_w(i) = 0.0;

    share ***s_X = (share***) malloc(sizeof(share**) * rows);
    for (int i = 0; i < rows; i++)
        s_X[i] = (share**) malloc(sizeof(share*) * columns);
    share **s_y = (share**) malloc(sizeof(share*) * rows);
    share **s_w = (share**) malloc(sizeof(share*) * columns);

    generate_shared_data (engine, s_X, s_y, s_w, plain_X, plain_y, plain_w);

    share **s_out = sgd (engine, params, s_X, s_w, s_y, indices, columns);

    for (int i = 0; i < columns; i++)
        s_out[i] = engine->ac->PutOUTGate(s_out[i], ALL);

    engine->party->ExecCircuit();

    Vector<double> plain_output = plain_sgd (params, plain_X, plain_y, plain_w, indices);

//    std::cout << model << "\n";  // Verify result for genModel

    uint32_t output[columns];
    std::cout << "---encrypted sgd---\n";
    for (int i = 0; i < columns; i++)
    {
        output[i] = s_out[i]->get_clear_value<uint32_t>();
        std::cout << "w[" << i << "] = "  << (int32_t)output[i] / (double) engine->precision;

        std::cout << "\n";
    }
    std::cout << "---plain sgd---\n";
    for (int i = 0; i < columns; i++)
        std::cout << "w[" << i << "] = "  << plain_output(i) << "\n";

    std::cout << "\n";

    for (int i = 0; i < rows; i++)
        free (s_X[i]);
    free (s_X);
    free (s_y);
    free (s_w);

    return 0;
}

