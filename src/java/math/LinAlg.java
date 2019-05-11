/*
 * Copyright 2019 Aman Mehara
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package math;

import java.util.function.Function;

public class LinAlg {

    public static double[][] broadcast(double[][] matrix, int factor) {

        assert matrix.length == 1 || matrix[0].length == 1;

        boolean broadcastOuterDim;

        var outerDim = matrix.length;
        var innerDim = matrix[0].length;

        if (outerDim == 1) {
            outerDim *= factor;
            broadcastOuterDim = true;
        } else {
            innerDim *= factor;
            broadcastOuterDim = false;
        }

        var m = new double[outerDim][innerDim];
        for (var outerIdx = 0; outerIdx < outerDim; outerIdx++) {
            for (var innerIdx = 0; innerIdx < innerDim; innerIdx++) {
                if (broadcastOuterDim) {
                    m[outerIdx][innerIdx] = matrix[0][innerIdx];
                } else {
                    m[outerIdx][innerIdx] = matrix[outerIdx][0];
                }
            }
        }
        return m;
    }

    public static double[][] matrixAddition(double[][] matrix1, double[][] matrix2) {

        assert matrix1.length == matrix2.length && matrix1[0].length == matrix2[0].length;

        var outerDim = matrix1.length;
        var innerDim = matrix1[0].length;

        var m = new double[outerDim][innerDim];
        for (var outerIdx = 0; outerIdx < outerDim; outerIdx++) {
            for (var innerIdx = 0; innerIdx < innerDim; innerIdx++) {
                m[outerIdx][innerIdx] = matrix1[outerIdx][innerIdx] + matrix2[outerIdx][innerIdx];
            }
        }
        return m;
    }

    public static double[][] matrixMultiplication(double[][] matrix1, double[][] matrix2) {

        assert matrix1[0].length == matrix2.length;

        var outerDim = matrix1.length;
        var innerDim = matrix2[0].length;

        var m = new double[outerDim][innerDim];
        for (var outerIdx = 0; outerIdx < matrix1.length; outerIdx++) {
            for (var innerIdx = 0; innerIdx < matrix2[0].length; innerIdx++) {
                m[outerIdx][innerIdx] = 0.0;
                for (var cumIndex = 0; cumIndex < matrix2.length; cumIndex++) {
                    m[outerIdx][innerIdx] += matrix1[outerIdx][cumIndex] * matrix2[cumIndex][innerIdx];
                }
            }
        }
        return m;

    }

    public static double[][] matrixTranspose(double[][] matrix) {

        var outerDim = matrix[0].length;
        var innerDim = matrix.length;

        var m = new double[outerDim][innerDim];
        for (var outerIdx = 0; outerIdx < outerDim; outerIdx++) {
            for (var innerIdx = 0; innerIdx < innerDim; innerIdx++) {
                m[outerIdx][innerIdx] = matrix[innerIdx][outerIdx];
            }
        }
        return m;

    }

    public static double[][] reshape(double[] vector, int outerDim, int innerDim) {

        assert vector.length == outerDim * innerDim;

        var matrix = new double[outerDim][innerDim];
        for (var idx = 0; idx < vector.length; idx++) {
            matrix[idx / innerDim][idx % innerDim] = vector[idx];
        }
        return matrix;

    }

    public static double[] reshape(double[][] matrix) {

        var outerDim = matrix.length;
        var innerDim = matrix[0].length;
        var dim = outerDim * innerDim;

        var vector = new double[dim];
        for (var idx = 0; idx < vector.length; idx++) {
            vector[idx] = matrix[idx / innerDim][idx % innerDim];
        }
        return vector;

    }

    public static double[][] scalarMultiplication(double[][] matrix, double scalar) {

        var outerDim = matrix.length;
        var innerDim = matrix[0].length;

        var m = new double[outerDim][innerDim];
        for (var outerIdx = 0; outerIdx < outerDim; outerIdx++) {
            for (var innerIdx = 0; innerIdx < innerDim; innerIdx++) {
                m[outerIdx][innerIdx] = matrix[outerIdx][innerIdx] * scalar;
            }
        }
        return m;
    }

    public static double[][] transform(double[][] matrix, Function<Double, Double> function) {

        var outerDim = matrix.length;
        var innerDim = matrix[0].length;

        var m = new double[outerDim][innerDim];
        for (var outerIdx = 0; outerIdx < outerDim; outerIdx++) {
            for (var innerIdx = 0; innerIdx < innerDim; innerDim++) {
                m[outerIdx][innerIdx] = function.apply(matrix[outerIdx][innerIdx]);
            }
        }

        return m;

    }

}
