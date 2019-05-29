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

package com.amanmehara.tantrika.nn;

import com.amanmehara.tantrika.math.linalg.Matrix;
import com.amanmehara.tantrika.math.linalg.Vector;
import com.amanmehara.tantrika.nn.activations.Activation;
import com.amanmehara.tantrika.nn.initializers.Initializer;

public class Layer {

    private final int units;
    private final int inputDimension;
    private final Activation activation;
    private final boolean useBias;

    private Matrix kernel;
    private Vector bias;

    public Layer(int units, int inputDimension, Activation activation, Initializer kernelInitializer) {
        this.units = units;
        this.inputDimension = inputDimension;
        this.activation = activation;
        this.useBias = false;

        this.kernel = kernelInitializer.initializeMatrix(units, inputDimension);
        this.bias = null;
    }

    public Layer(int units, int inputDimension, Activation activation, Initializer kernelInitializer, Initializer biasInitializer) {
        this.units = units;
        this.inputDimension = inputDimension;
        this.activation = activation;
        this.useBias = true;

        this.kernel = kernelInitializer.initializeMatrix(units, inputDimension);
        this.bias = biasInitializer.initializeVector(units);
    }

    public int units() {
        return units;
    }

    public int inputDimension() {
        return inputDimension;
    }

    public Activation activation() {
        return activation;
    }

    public boolean useBias() {
        return useBias;
    }

    public Matrix kernel() {
        return kernel;
    }

    public void kernel(Matrix kernel) {
        if (this.kernel.outerSize() != kernel.outerSize() || this.kernel.innerSize() != kernel.innerSize()) {
            throw new IllegalArgumentException();
        }

        this.kernel = kernel;
    }

    public Vector bias() {
        return bias;
    }

    public void bias(Vector bias) {
        if (!useBias || this.bias.size() != bias.size()) {
            throw new IllegalArgumentException();
        }

        this.bias = bias;
    }

    public Matrix computeOutputs(Matrix inputs) {
        var outputs = kernel.multiply(inputs);
        if (useBias) {
            var factor = inputs.innerSize();
            outputs = outputs.add(bias.broadcast(1, factor));
        }
        return outputs.transform(activation::value);
    }

}
