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
}

struct RegressionParams
{
    uint32_t maxIterations;
    double   learningRate;
};

typedef struct
{
    ABYParty *party;
    ArithmeticCircuit *ac;
    BooleanCircuit *bc;
    BooleanCircuit *yc;
    e_role role;
    uint32_t precision;
    uint32_t shift;
} EncryptionEngine;

int32_t main_sgd (e_role role, double learning_rate, uint32_t precision,
                  uint32_t max_iter, bool provide_data, const std::string& address,
                  uint16_t port, seclvl seclvl, uint32_t nvals, uint32_t nthreads,
                  e_mt_gen_alg mt_alg);

#endif /* __SGD_H_ */
