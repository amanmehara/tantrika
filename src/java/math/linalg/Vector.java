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

    private final double[] v;
    private final int size;

    public Vector(double[] vector) {
        v = vector;
        size = vector.length;
    }

    public Vector(int size) {
        v = new double[size];
        this.size = size;
    }

    public Vector(int size, Supplier<Double> supplier) {
        v = new double[size];
        this.size = size;

        for (var index = 0; index < size; index++) {
            v[index] = supplier.get();
        }
    }

    public int size() {
        return size;
    }

    public double get(int index) {
        return v[index];
    }

    public Matrix broadcast(int dimension, int factor) {
        if (dimension != 1 && dimension != 2) {
            throw new IllegalArgumentException();
        }

        var outerSize = dimension == 2 ? factor : size;
        var innerSize = dimension == 1 ? factor : size;

        var m = new double[outerSize][innerSize];
        for (var outerIndex = 0; outerIndex < outerSize; outerIndex++) {
            for (var innerIndex = 0; innerIndex < innerSize; innerIndex++) {
                if (dimension == 2) {
                    m[outerIndex][innerIndex] = v[innerIndex];
                } else {
                    m[outerIndex][innerIndex] = v[outerIndex];
                }
            }
        }
        return new Matrix(m);
    }

    public Matrix reshape(int outerSize, int innerSize) {
        if (size != outerSize * innerSize) {
            throw new IllegalArgumentException();
        }

        var m = new double[outerSize][innerSize];
        for (var index = 0; index < size; index++) {
            m[index / innerSize][index % innerSize] = v[index];
        }
        return new Matrix(m);
    }

}
