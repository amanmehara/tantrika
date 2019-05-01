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

public class LinAlg {

    public static double[][] matmul(double[][] matrix1, double[][] matrix2) {

        assert matrix1[0].length == matrix2.length;

        int outerDim = matrix1.length;
        int innerDim = matrix2[0].length;

        double[][] matrix = new double[outerDim][innerDim];
        for (int outerIdx = 0; outerIdx < matrix1.length; outerIdx++) {
            for (int innerIdx = 0; innerIdx < matrix2[0].length; innerIdx++) {
                matrix[outerIdx][innerIdx] = 0.0;
                for (int cumIndex = 0; cumIndex < matrix2.length; cumIndex++) {
                    matrix[outerIdx][innerIdx] += matrix[outerIdx][cumIndex] * matrix[cumIndex][innerIdx];
                }
            }
        }
        return matrix;

    }

    public static double[][] reshape(double[] vector, int outerDim, int innerDim) {

        assert vector.length == outerDim * innerDim;

        double[][] matrix = new double[outerDim][innerDim];
        for (int idx = 0; idx < vector.length; idx++) {
            matrix[idx / outerDim][idx % innerDim] = vector[idx];
        }
        return matrix;

    }

}
