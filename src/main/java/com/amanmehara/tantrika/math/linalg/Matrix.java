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

package com.amanmehara.tantrika.math.linalg;

import java.util.function.DoubleSupplier;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

public class Matrix {

    private final double[] array;
    private final int outerSize;
    private final int innerSize;
    private final int size;

    public Matrix(final double[][] matrix) {
        if (matrix.length == 0) {
            throw new IllegalArgumentException();
        }

        outerSize = matrix.length;
        innerSize = matrix[0].length;
        size = outerSize * innerSize;

        boolean isInvalid = IntStream
                .range(0, matrix.length)
                .anyMatch(value -> innerSize != matrix[value].length);

        if (isInvalid) {
            throw new IllegalArgumentException();
        }


        array = new double[size];

        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            System.arraycopy(matrix[outerIndex], 0, array, outerIndex * innerSize, innerSize);
        }
    }

    public Matrix(final int outerSize, final int innerSize) {
        this(outerSize, innerSize, () -> 0.0);
    }

    public Matrix(final int outerSize, final int innerSize, final DoubleSupplier supplier) {
        this.outerSize = outerSize;
        this.innerSize = innerSize;
        size = outerSize * innerSize;
        array = new double[size];

        for (var index = 0; index < size; index++) {
            array[index] = supplier.getAsDouble();
        }
    }

    public Matrix(final int outerSize, final int innerSize, final double[] array) {
        var size = outerSize * innerSize;

        if (size != array.length) {
            throw new IllegalArgumentException();
        }

        this.outerSize = outerSize;
        this.innerSize = innerSize;
        this.size = size;
        this.array = array;
    }

    public int outerSize() {
        return outerSize;
    }

    public int innerSize() {
        return innerSize;
    }

    public double get(int outerIndex, int innerIndex) {
        return array[outerIndex * innerSize + innerIndex];
    }

    public Matrix add(final Matrix matrix) {
        if (outerSize != matrix.outerSize || innerSize != matrix.innerSize) {
            throw new IllegalArgumentException();
        }

        var sum = new Matrix(outerSize, innerSize);
        for (var index = 0; index < size; index++) {
            sum.array[index] = array[index] + matrix.array[index];
        }
        return sum;
    }

    public Matrix hadamardProduct(Matrix matrix) {
        if (outerSize != matrix.outerSize || innerSize != matrix.innerSize) {
            throw new IllegalArgumentException();
        }

        var hadamardProduct = new Matrix(outerSize, innerSize);
        for (var index = 0; index < size; index++) {
            hadamardProduct.array[index] = array[index] * matrix.array[index];
        }
        return hadamardProduct;
    }

    // TODO: Use Strassen's matrix multiplication algorithm.
    public Matrix multiply(final Matrix matrix) {
        if (innerSize != matrix.outerSize) {
            throw new IllegalArgumentException();
        }

        var product = new Matrix(outerSize, matrix.innerSize);
        for (var outerIndex = 0; outerIndex < product.outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < product.innerSize; innerIndex++) {
                var index = outerIndex * product.innerSize + innerIndex;
                product.array[index] = 0.0;
                for (var cumIndex = 0; cumIndex < innerSize; cumIndex++) {
                    product.array[index] += get(outerIndex, cumIndex) * matrix.get(cumIndex, innerIndex);
                }
            }
        }
        return product;

    }

    public Vector reshape() {
        return new Vector(array);
    }

    public Matrix scale(final double factor) {
        var scale = new Matrix(outerSize, innerSize);

        for (var index = 0; index < size; index++) {
            scale.array[index] = array[index] * factor;
        }
        return scale;
    }

    public Matrix transform(final DoubleUnaryOperator function) {
        var transform = new Matrix(outerSize, innerSize);
        for (var index = 0; index < size; index++) {
            transform.array[index] = function.applyAsDouble(array[index]);
        }
        return transform;
    }

    public Matrix transpose() {
        var transpose = new Matrix(innerSize, outerSize);
        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                var transposeIndex = innerIndex * outerSize + outerIndex;
                transpose.array[transposeIndex] = get(outerIndex, innerIndex);
            }
        }
        return transpose;
    }
}
