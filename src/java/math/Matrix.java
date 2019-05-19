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
import java.util.stream.IntStream;

public class Matrix {

    private final double[][] m;
    private final int outerDimension;
    private final int innerDimension;

    public Matrix(final int outerDimension, final int innerDimension) {
        m = new double[outerDimension][innerDimension];
        this.outerDimension = outerDimension;
        this.innerDimension = innerDimension;
    }

    public Matrix(final double[][] matrix) {
        if (matrix.length == 0) {
            throw new IllegalArgumentException();
        }

        m = matrix;
        outerDimension = matrix.length;
        innerDimension = matrix[0].length;

        boolean isInvalid = IntStream
                .range(0, matrix.length)
                .anyMatch(value -> innerDimension != matrix[value].length);

        if (isInvalid) {
            throw new IllegalArgumentException();
        }
    }

    public Matrix add(final Matrix matrix) {
        if (outerDimension != matrix.outerDimension || innerDimension != matrix.innerDimension) {
            throw new IllegalArgumentException();
        }

        var sum = new Matrix(outerDimension, innerDimension);
        for (var outerIndex = 0; outerIndex < outerDimension; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerDimension; innerIndex++) {
                sum.m[outerIndex][innerIndex] = m[outerIndex][innerIndex] + matrix.m[outerIndex][innerIndex];
            }
        }
        return sum;
    }

    // TODO: Use Strassen's matrix multiplication algorithm.
    public Matrix multiply(final Matrix matrix) {
        if (innerDimension != matrix.outerDimension) {
            throw new IllegalArgumentException();
        }

        var product = new Matrix(outerDimension, matrix.innerDimension);
        for (var outerIndex = 0; outerIndex < product.outerDimension; outerIndex++) {
            for (var innerIndex = 0; innerIndex < product.innerDimension; innerIndex++) {
                product.m[outerIndex][innerIndex] = 0.0;
                for (var cumIndex = 0; cumIndex < innerDimension; cumIndex++) {
                    product.m[outerIndex][innerIndex] += m[outerIndex][cumIndex] * matrix.m[cumIndex][innerIndex];
                }
            }
        }
        return product;

    }

    public Matrix scale(final double factor) {
        var scale_ = new Matrix(outerDimension, innerDimension);
        for (var outerIndex = 0; outerIndex < outerDimension; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerDimension; innerIndex++) {
                scale_.m[outerIndex][innerIndex] = m[outerIndex][innerIndex] * factor;
            }
        }
        return scale_;
    }

    public Matrix transform(Function<Double, Double> function) {
        var transform = new Matrix(outerDimension, innerDimension);
        for (var outerIndex = 0; outerIndex < outerDimension; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerDimension; innerIndex++) {
                transform.m[outerIndex][innerIndex] = function.apply(m[outerIndex][innerIndex]);
            }
        }
        return transform;
    }

    public Matrix transpose() {
        var transpose = new Matrix(innerDimension, outerDimension);
        for (var outerIndex = 0; outerIndex < outerDimension; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerDimension; innerIndex++) {
                transpose.m[innerIndex][outerIndex] = m[outerIndex][innerIndex];
            }
        }
        return transpose;
    }
}
