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

package nn;

import math.linalg.Matrix;
import math.linalg.Vector;
import nn.activations.Activation;

import java.util.Random;
import java.util.function.Supplier;

public class Layer {

    private final int units;
    private final int inputDimension;
    private final Activation activation;
    private final boolean useBias;

    private Matrix kernel;
    private Vector bias;

    public Layer(int units, int inputDimension, Activation activation, boolean useBias) {
        this.units = units;
        this.inputDimension = inputDimension;
        this.activation = activation;
        this.useBias = useBias;

        this.kernel = this.initializeKernel();
        this.bias = useBias ? this.initializeBias() : null;
    }

    // TODO: Use Initializers.
    private Matrix initializeKernel() {
        Random random = new Random();
        Supplier<Double> randomSupplier = () -> random.nextDouble() * 2.0 - 1.0;
        return new Matrix(units, inputDimension, randomSupplier);
    }

    // TODO: Use Initializers.
    private Vector initializeBias() {
        return new Vector(units);
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
        if(this.kernel.outerSize() != kernel.outerSize() || this.kernel.innerSize() != kernel.innerSize()) {
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
