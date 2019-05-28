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

import java.util.function.Supplier;

public class Vector {

    private final Double[] array;
    private final int size;

    public Vector(Double[] vector) {
        array = vector;
        size = vector.length;
    }

    public Vector(int size) {
        this(size, () -> 0.0);
    }

    public Vector(int size, Supplier<Double> supplier) {
        array = new Double[size];
        this.size = size;

        for (var index = 0; index < size; index++) {
            array[index] = supplier.get();
        }
    }

    public int size() {
        return size;
    }

    public double get(int index) {
        return array[index];
    }

    public Matrix broadcast(int dimension, int factor) {
        if (dimension != 1 && dimension != 2) {
            throw new IllegalArgumentException();
        }

        var outerSize = dimension == 2 ? factor : size;
        var innerSize = dimension == 1 ? factor : size;

        var m = new Double[outerSize][innerSize];
        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                if (dimension == 2) {
                    m[outerIndex][innerIndex] = array[innerIndex];
                } else {
                    m[outerIndex][innerIndex] = array[outerIndex];
                }
            }
        }
        return new Matrix(m);
    }

    public Matrix reshape(int outerSize, int innerSize) {
        return new Matrix(outerSize, innerSize, array);
    }

}
