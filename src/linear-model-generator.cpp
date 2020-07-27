/**
 \file      linear-model-generator.cpp
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

#include "linear-model-generator.h"

#include <random>

namespace encsgd
{
    void LinearModelGen::setModel(Vector<double>& model, double noise, double sd)
    {
        model_w = model;
        mean = noise;
        standard_deviation = sd;
    }

    void LinearModelGen::sample(Matrix<double>& X, Vector<double>& y)
    {
        if (X.rows() != y.size()) throw std::runtime_error("error in LinearModelGen::sample");
        if (X.cols() != model_w.size()) throw std::runtime_error("error in LinearModelGen::sample");

        std::default_random_engine generator;
        std::normal_distribution<double> distribution(mean, standard_deviation);

        Vector<double> noise(X.rows());

        for (int i = 0; i < X.rows(); i++)
        {
            for (int j = 0; j < X.cols(); j++)
                X(i, j) = distribution(generator);

            noise(i) = distribution(generator);
        }

        y = X * model_w + noise;
    }
}
