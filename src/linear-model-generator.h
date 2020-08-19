/**
 \file      linear-model-generator.h
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
 \brief     Linear Model Generator for Stochastic Gradient Descent using the ABY framework.
 */

#ifndef LINEAR_MODEL_GENERATOR_H
#define LINEAR_MODEL_GENERATOR_H

#include "../extern/eigen/Eigen/Dense"
#include "sgd.h"

namespace encsgd
{
    class LinearModelGen
    {
        public:

            Vector<double> model_w;
            double mean, standard_deviation;

            void setModel(Vector<double>& model, double noise = 1, double sd = 1);

            void sample(Matrix<double>& X, Vector<double>& y);
    };

    class LogisticModelGen
    {
        public:

            Vector<double> model_w;
            double mean, standard_deviation;

            void setModel(Vector<double>& model, double noise = 1, double sd = 1);

            void sample(Matrix<double>& X, Vector<double>& y, bool print);
    };
}

#endif
