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
#include "linear-model-generator.h"

#define INPUT_BITLEN  32

using namespace encsgd;


sgd::sgd (e_role role, uint32_t precision, const std::string& address,
          uint16_t port, seclvl seclvl, uint32_t nthreads, e_mt_gen_alg mt_alg)
{
    uint32_t bitlen = INPUT_BITLEN;

    mParty = new ABYParty(role, address, port, seclvl, bitlen, nthreads, mt_alg);

    std::vector<Sharing*>& sharing = mParty->GetSharings();

    mRole = role;

    mPrecision = 1ull << precision;

    mShift = precision;

    mArithCir = (ArithmeticCircuit*) sharing[S_ARITH]->GetCircuitBuildRoutine();

    mBoolCir = (BooleanCircuit*) sharing[S_BOOL]->GetCircuitBuildRoutine();

    mYaoCir = (BooleanCircuit*) sharing[S_YAO]->GetCircuitBuildRoutine();
}

static void
get_plain_data (Matrix<double>& plain_X, Vector<double>& plain_y)
{
    /* You can insert your according data and label files here */

    std::ifstream features("../data/train.data");
    for (int i = 0; i < plain_X.rows(); i++)
    {
        for (int j = 0; j < plain_X.cols(); j++)
            features >> plain_X(i,j);
        features >> plain_y(i);
    }
    features.close();

    std::ifstream labels("../data/train.labels");
    for (int i = 0; i < plain_X.rows(); i++)
    {
        labels >> plain_y(i);
    }
    labels.close();
}

void
sgd::generate_shared_data (share ***s_X, share **s_y, share **s_w,
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

    plain_X *= mPrecision;
    X = plain_X.cast <uint32_t> ();
    X_1.setRandom (rows, columns);
    X_0 = X - X_1;

    plain_y *= mPrecision;
    y = plain_y.cast <uint32_t> ();
    y_1.setRandom (rows);
    y_0 = y - y_1;

    plain_w *= mPrecision;
    w = plain_w.cast <uint32_t> ();
    w_1.setRandom (columns);
    w_0 = w - w_1;

    if (mRole == SERVER)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                s_X[i][j] = mArithCir->PutSharedINGate(X_0(i, j), INPUT_BITLEN);

        for (int i = 0; i < rows; i++)
            s_y[i] = mArithCir->PutSharedINGate(y_0(i), INPUT_BITLEN);

        for (int i = 0; i < columns; i++)
            s_w[i] = mArithCir->PutSharedINGate(w_0(i), INPUT_BITLEN);
    }
    else
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                s_X[i][j] = mArithCir->PutSharedINGate(X_1(i, j), INPUT_BITLEN);

        for (int i = 0; i < rows; i++)
            s_y[i] = mArithCir->PutSharedINGate(y_1(i), INPUT_BITLEN);

        for (int i = 0; i < columns; i++)
            s_w[i] = mArithCir->PutSharedINGate(w_1(i), INPUT_BITLEN);
    }
}

void
sgd::get_data (RegressionParams params,
               Matrix<double>& plain_X, Vector<double>& plain_y, Vector<double>& plain_w,
               bool provide_data)
{
    if (provide_data)
        get_plain_data (plain_X, plain_y);
    else
    {   // Generate artificial data
        LinearModelGen gen;

        Vector<double> model(plain_w.size());
        for (int i = 0; i < model.size(); ++i)
        {
            model(i) = rand() * i % 10;
        }
        gen.setModel(model);

        gen.sample(plain_X, plain_y);
    }

    for (uint32_t i = 0; i < params.maxIterations; i++)
    {
        params.indices[i] = i % plain_X.rows(); // FIXME: indices[i] = rand() % rows; leads to silly results.
                                         // Why? It seems rand() does interfere with
                                         // something from ABY...
    }
    for (int i = 0; i < plain_w.size(); i++)
        plain_w(i) = 1.0;

}

/* This is a very inefficient hack which simulates mul_truncation which is not available in ABY */
/* Returns z = ina*inb */
share*
sgd::mul_trunc (share *ina, share *inb)
{
    share *res, *res_is_less_zero, *res_inv;

    res = mArithCir->PutMULGate(ina, inb);

    res = mBoolCir->PutA2BGate(res, mYaoCir);

    res_is_less_zero = mBoolCir->PutGTGate(res, s_mThreshold);
    res_inv = mBoolCir->PutINVGate(res);
    res = mBoolCir->PutBarrelRightShifterGate(res, s_mShift);
    res_inv = mBoolCir->PutBarrelRightShifterGate(res_inv, s_mShift);
    res_inv = mBoolCir->PutINVGate(res_inv);
    res = mBoolCir->PutMUXGate(res_inv, res, res_is_less_zero);

    res = mArithCir->PutB2AGate(res);

    return res;
}

share*
sgd::inner_prod(share **ina, share **inb, int columns)
{
    share **temp = (share**) malloc (sizeof(share*) * columns);

    // pairwise multiplication of all input values
    for (int i = 0; i < columns; i++)
    {
        temp[i] = mul_trunc(ina[i], inb[i]);
    }

    // add up the individual multiplication results
    // in arithmetic sharing ADD is for free, and does not add circuit depth, thus simple sequential adding
    for (int i = 1; i < columns; i++)
    {
        temp[0] = mArithCir->PutADDGate(temp[0], temp[i]);
    }

    return temp[0];
}

share**
sgd::build_esgd_circuit (RegressionParams params,
                         share ***s_X, share **s_w, share **s_y, int columns)
{
    share *error;
    share **grad = (share**) malloc (sizeof(share*) * columns);

    uint32_t learning_rate = (uint32_t) (params.learningRate * mPrecision);

    for (uint32_t k = 0; k < params.maxIterations; k++)
    {
        /* We setup the formula: [df(w)_i]l = x_i_l * (w*x - y)
         * i := indices[k]
         * In the following comments x:= x_i
         */

        // compute x*w
        error = inner_prod(s_X[params.indices[k]], s_w, columns);
//        mArithCir->PutPrintValueGate(error, "xw");

        // compute x*w-y
        error = mArithCir->PutSUBGate(error, s_y[params.indices[k]]);
//        mArithCir->PutPrintValueGate(temp[0], "xw-y");

        share *alpha = mArithCir->PutCONSGate(learning_rate, INPUT_BITLEN);
        for (int i = 0; i < columns; i++)
        {
            // compute x_l(x*w-y)
            grad[i] = mul_trunc (error, s_X[params.indices[k]][i]);
//            mArithCir->PutPrintValueGate(temp[i], "x_l(x*w-y)");
            // update w = w - alpha * x_l(x*w-y)
            s_w[i] = mArithCir->PutSUBGate(s_w[i], mul_trunc (alpha, grad[i]));
//            mArithCir->PutPrintValueGate(s_w[i], "w=w-alpha*x_l(x*w-y)");
        }
    }

    for (int i = 0; i < columns; i++)
        free (grad[i]);
    free (grad);

    return s_w;
}

double
empirical_risk (Matrix<double> X, Vector<double> y, Vector<double> w)
{
    Vector<double> error =  X * w - y;

    return error.norm();
}

Vector<double>
sgd::plain_linear_regression (RegressionParams params,
                              Matrix<double> X, Vector<double> y, Vector<double> w)
{
    Vector<double> grad;
    double error;
    double alpha = params.learningRate;

    for (uint32_t k = 0; k < params.maxIterations; k++)
    {
            if (k % 10 == 0)
                std::cout << "--- risk: " << empirical_risk (X, y, w) << "---\n";

            error = X.row(params.indices[k]).dot(w);
            error -= y(params.indices[k]);
            grad = error * X.row(params.indices[k]);
            w = w - alpha * grad;
    }

    return w;
}

std::array<double,2>
test_plain_logistic_regression (Matrix<double> X, Vector<double> y, Vector<double> w)
{
    Vector<double> xw =  X * w;
    Vector<double> ret(xw.size());
    uint64_t count = 0;

    for (int i = 0; i < xw.size(); i++)
        ret(i) = 1.0 / (1 + std::exp(-xw(i)));

    ret -= y;

	for (int i = 0; i < xw.size(); ++i)
	{
		bool c0 = xw(i) > 0.5;
		bool c1 = y(i) > 0.5;

		count += (c0 == c1);
	}

    return {ret.norm(), (double) count };
}

Vector<double>
sgd::plain_logistic_regression (RegressionParams params,
                                Matrix<double> X, Vector<double> y, Vector<double> w)
{
    Vector<double> grad;
    double xw, sxw;
    double alpha = params.learningRate;

    for (uint32_t k = 0; k < params.maxIterations; k++)
    {
            if (k % 10 == 0)
            {
                std::array<double,2> r = test_plain_logistic_regression (X, y, w);
                std::cout << "--- risk: " << r[0] << "---\n";
            }

            xw = X.row(params.indices[k]).dot(w);
            sxw = 1.0 / (1 + std::exp(-xw));
            grad = (sxw - y(params.indices[k])) * X.row(params.indices[k]);
            w = w - alpha * grad;
    }

    return w;
}

uint32_t*
sgd::encrypted_sgd (RegressionParams params,
                    Matrix<double> plain_X, Vector<double> plain_y, Vector<double> plain_w)
{
    int rows = plain_X.rows();
    int columns = plain_X.cols();

    share ***s_X = (share***) malloc(sizeof(share**) * rows);
    for (int i = 0; i < rows; i++)
        s_X[i] = (share**) malloc(sizeof(share*) * columns);
    share **s_y = (share**) malloc(sizeof(share*) * rows);
    share **s_w = (share**) malloc(sizeof(share*) * columns);

    generate_shared_data (s_X, s_y, s_w, plain_X, plain_y, plain_w);

    // Put in Constants
    s_mThreshold = mBoolCir->PutCONSGate(1ull << (INPUT_BITLEN-1), INPUT_BITLEN);
    s_mShift = mBoolCir->PutCONSGate(mShift, INPUT_BITLEN);

    share **s_out = build_esgd_circuit (params, s_X, s_w, s_y, columns);

    /**
     Step 8: Output the value of s_out (the computation result) to both parties
     */
    for (int i = 0; i < columns; i++)
        s_out[i] = mArithCir->PutOUTGate(s_out[i], ALL);

    /**
     Step 9: Executing the circuit using the ABYParty object evaluate the
     problem.
     */
    mParty->ExecCircuit();

    /**
     Step 10: Type cast the plaintext output to 32 bit unsigned integer.
     */
    uint32_t *output = (uint32_t*) malloc (sizeof(uint32_t) * columns);
    for (int i = 0; i < columns; i++)
    {
        output[i] = s_out[i]->get_clear_value<uint32_t>();
    }

    // free memory
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
            free (s_X[i][j]);
        free (s_X[i]);
    }
    free (s_X);
    for (int i = 0; i < rows; i++)
        free (s_y[i]);
    free (s_y);
    for (int i = 0; i < columns; i++)
        free (s_w[i]);
    free (s_w);

    return output;
}

