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

import java.util.Arrays;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

public class Vector {

    private final double[] array;
    private final int size;

    public Vector(double[] vector) {
        array = vector;
        size = vector.length;
    }

    public Vector(int size) {
        this(size, () -> 0.0);
    }

    public Vector(int size, DoubleSupplier supplier) {
        array = new double[size];
        this.size = size;

        for (var index = 0; index < size; index++) {
            array[index] = supplier.getAsDouble();
        }
    }

    public int size() {
        return size;
    }

    public double get(int index) {
        return array[index];
    }

    public Matrix broadcast(int dimension, int factor) {


        var matrixSize = size * factor;
        var matrix = new double[matrixSize];

        int outerSize;
        int innerSize;

        switch (dimension) {
            case 1:
                outerSize = this.size;
                innerSize = factor;
                IntStream.range(0, this.size).forEach(index -> {
                    Arrays.fill(matrix, index * factor, (index + 1) * factor, array[index]);
                });
                break;
            case 2:
                outerSize = factor;
                innerSize = this.size;
                IntStream.range(0, this.size).forEach(index -> {
                    System.arraycopy(array, 0, matrix, index * factor, factor);
                });
                break;
            default:
                throw new IllegalArgumentException();

        }

        return new Matrix(outerSize, innerSize, matrix);
    }

    public Matrix reshape(int outerSize, int innerSize) {
        return new Matrix(outerSize, innerSize, array);
    }

}
