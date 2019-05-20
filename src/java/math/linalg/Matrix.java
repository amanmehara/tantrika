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

package math.linalg;

import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class Matrix {

    private final double[][] m;
    private final int outerSize;
    private final int innerSize;

    public Matrix(final double[][] matrix) {
        if (matrix.length == 0) {
            throw new IllegalArgumentException();
        }

        m = matrix;
        outerSize = matrix.length;
        innerSize = matrix[0].length;

        boolean isInvalid = IntStream
                .range(0, matrix.length)
                .anyMatch(value -> innerSize != matrix[value].length);

        if (isInvalid) {
            throw new IllegalArgumentException();
        }
    }

    public Matrix(final int outerSize, final int innerSize) {
        m = new double[outerSize][innerSize];
        this.outerSize = outerSize;
        this.innerSize = innerSize;
    }

    public Matrix(final int outerSize, final int innerSize, final Supplier<Double> supplier) {
        m = new double[outerSize][innerSize];
        this.outerSize = outerSize;
        this.innerSize = innerSize;

        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                m[outerIndex][innerIndex] = supplier.get();
            }
        }
    }

    public int outerSize() {
        return outerSize;
    }

    public int innerSize() {
        return innerSize;
    }

    public double get(int outerIndex, int innerIndex) {
        return m[outerIndex][innerIndex];
    }

    public Matrix add(final Matrix matrix) {
        if (outerSize != matrix.outerSize || innerSize != matrix.innerSize) {
            throw new IllegalArgumentException();
        }

        var sum = new Matrix(outerSize, innerSize);
        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                sum.m[outerIndex][innerIndex] = m[outerIndex][innerIndex] + matrix.m[outerIndex][innerIndex];
            }
        }
        return sum;
    }

    // TODO: Use Strassen's matrix multiplication algorithm.
    public Matrix multiply(final Matrix matrix) {
        if (innerSize != matrix.outerSize) {
            throw new IllegalArgumentException();
        }

        var product = new Matrix(outerSize, matrix.innerSize);
        for (var outerIndex = 0; outerIndex < product.outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < product.innerSize; innerIndex++) {
                product.m[outerIndex][innerIndex] = 0.0;
                for (var cumIndex = 0; cumIndex < innerSize; cumIndex++) {
                    product.m[outerIndex][innerIndex] += m[outerIndex][cumIndex] * matrix.m[cumIndex][innerIndex];
                }
            }
        }
        return product;

    }

    public Vector reshape() {
        var size = outerSize * innerSize;
        var v = new double[size];
        for (var index = 0; index < size; index++) {
            v[index] = m[index / innerSize][index % innerSize];
        }
        return new Vector(v);
    }

    public Matrix scale(final double factor) {
        var scale = new Matrix(outerSize, innerSize);
        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                scale.m[outerIndex][innerIndex] = m[outerIndex][innerIndex] * factor;
            }
        }
        return scale;
    }

    public Matrix transform(final Function<Double, Double> function) {
        var transform = new Matrix(outerSize, innerSize);
        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                transform.m[outerIndex][innerIndex] = function.apply(m[outerIndex][innerIndex]);
            }
        }
        return transform;
    }

    public Matrix transpose() {
        var transpose = new Matrix(innerSize, outerSize);
        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                transpose.m[innerIndex][outerIndex] = m[outerIndex][innerIndex];
            }
        }
        return transpose;
    }
}
