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
 \brief     Stochastic Gradient Descent Test class implementation.
 */

//Utility libs
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
//ABY Party class
#include "../extern/ABY/src/abycore/aby/abyparty.h"

#include "sgd.h"

int32_t read_test_options(int32_t* argcp, char*** argvp, e_role* role,
                          double* learning_rate, uint32_t* precision,
                          uint32_t* max_iter, bool* provide_data, uint32_t* nvals,
                          uint32_t* secparam, std::string* address, uint16_t* port)
{

    uint32_t int_role = 0, int_port = 0, int_precision = 0;

    parsing_ctx options[] = {
        {(void*) &int_role, T_NUM, "r", "Role: 0/1", true, false },
        {(void*) learning_rate, T_DOUBLE, "l", "Learning Rate", true, true },

        {(void*) &int_precision, T_NUM, "p", "Precision, default: 8", false, false },
        {(void*) provide_data, T_FLAG, "d", "Provide Data, default: False", false, false },
        {(void*) max_iter, T_NUM, "t", "Max iterations, default: 100", false, false }, // FIXME: doesnt work yet

        {(void*) nvals, T_NUM, "n", "Number of parallel elements", false, false },
        {(void*) secparam, T_NUM, "s", "Symmetric Security Bits, default: 128", false, false },
        {(void*) address, T_STR, "a", "IP-address, default: localhost", false, false },
        {(void*) &int_port, T_NUM, "P", "Port, default: 7766", false, false }
    };

    if (!parse_options(argcp, argvp, options,
            sizeof(options) / sizeof(parsing_ctx))) {
        print_usage(*argvp[0], options, sizeof(options) / sizeof(parsing_ctx));
        std::cout << "Exiting" << std::endl;
        exit(0);
    }

    assert(int_role < 2);
    *role = (e_role) int_role;

    if (int_port != 0) {
        assert(int_port < 1 << (sizeof(uint16_t) * 8));
        *port = (uint16_t) int_port;
    }

    if (int_precision != 0)
        *precision = int_precision;

    if (role == 0 && (*learning_rate <= 0 || *learning_rate >= 1))
    {
        std::cout << "error: specify a valid learning rate (-l) in (0, 1)!" << std::endl;
        std::cout << "Exiting" << std::endl;
        exit(0);
    }

    if (role == 0 && (*learning_rate < 1.0 / (1ull << *precision)))
    {
        std::cout << "error: learning rate is too small (< 2^-Precision)" << std::endl;
        std::cout << "Exiting" << std::endl;
        exit(0);
    }

    return 1;
}

int main(int argc, char** argv)
{
    e_role role;
    double learning_rate = 0;
    uint32_t nvals = 1, secparam = 128, nthreads = 1, precision = 8, max_iter = 100;
    uint16_t port = 7766;
    bool provide_data = false;
    std::string address = "127.0.0.1";
    e_mt_gen_alg mt_alg = MT_OT;

    read_test_options (&argc, &argv, &role, &learning_rate, &precision, &max_iter,
                       &provide_data, &nvals, &secparam, &address, &port);

    seclvl seclvl = get_sec_lvl(secparam);

    main_sgd (role, learning_rate, precision, max_iter, provide_data, address,
              port, seclvl, nvals, nthreads, mt_alg);

    return 0;
}

