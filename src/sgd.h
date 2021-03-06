/**
 \file      sgd.h
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

#ifndef __SGD_H_
#define __SGD_H_

#include "../extern/ABY/src/abycore/circuit/booleancircuits.h"
#include "../extern/ABY/src/abycore/circuit/arithmeticcircuits.h"
#include "../extern/ABY/src/abycore/circuit/circuit.h"
#include "../extern/ABY/src/abycore/aby/abyparty.h"
#include "../extern/eigen/Eigen/Dense"
#include "../extern/eigen/Eigen/Core"
#include <math.h>
#include <cassert>

namespace encsgd
{
    template<typename T>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template<typename T>
    using Vector = Eigen::Vector<T, Eigen::Dynamic>;

    struct RegressionParams
    {
        uint32_t maxIterations;
        double   learningRate;
        uint32_t *indices;
    };

    class sgd
    {
        public:

        void get_data (RegressionParams params, Matrix<double>& X, Vector<double>& y, Vector<double>& w, bool provide_data);

        // TODO: Move plain_sgd to a seperate class
        Vector<double> plain_linear_regression (RegressionParams params, Matrix<double> X, Vector<double> y, Vector<double> w);

        Vector<double> plain_logistic_regression (RegressionParams params, Matrix<double> X, Vector<double> y, Vector<double> w);

        Vector<uint32_t> encrypted_linear_regression (RegressionParams params, Matrix<double> X, Vector<double> y, Vector<double> w);

        Vector<uint32_t>
        encrypted_logistic_regression (RegressionParams params,
                    Matrix<double> plain_X, Vector<double> plain_y, Vector<double> plain_w);

        sgd (e_role role, uint32_t precision, const std::string& address, uint16_t port, seclvl seclvl, uint32_t nthreads, e_mt_gen_alg mt_alg);

        uint32_t get_precision () { return mPrecision;};

        private:

        ABYParty *mParty;
        ArithmeticCircuit *mArithCir;
        BooleanCircuit *mBoolCir;
        BooleanCircuit *mYaoCir;
        e_role mRole;
        uint32_t mPrecision;
        uint32_t mShift;
        share *s_mThreshold;
        share *s_mShift;
        share *s_mFour;
        share *s_mF;
        share *s_mZero;
        share *s_mOne;
        share *s_mThresholdYao;

        void
        generate_shared_data (share ***s_X, share **s_y, share **s_w,
                              Matrix<double> plain_X, Vector<double> plain_y, Vector<double> plain_w);

        share*
        mul_trunc (share *ina, share *inb);

        double
        activation_function(double xw);

        share*
        activation_function(share *u);

        share*
        inner_prod(share **ina, share **inb, int columns);

        share**
        build_linear_regression_circuit (RegressionParams params, share ***s_X,
                            share **s_w, share **s_y, int columns);

        share**
        build_logistic_regression_circuit (RegressionParams params,
                         share ***s_X, share **s_w, share **s_y, int columns);

    };
}

#endif /* __SGD_H_ */
