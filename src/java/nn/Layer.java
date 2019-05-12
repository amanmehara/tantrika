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

import math.LinAlg;
import nn.activations.Activation;

import java.util.Random;

public class Layer {

    private final Activation activation;
    public double[] bias;
    public double[][] kernel;
    private final int inputDim;
    public final int units;
    private final boolean useBias;

    public double[] biasDifference;
    public double[][] kernelDifference;

    public double[][] signalError;

    public double[][] inputs;

    public Layer(int units, int inputDim, Activation activation, boolean useBias) {
        this.units = units;
        this.inputDim = inputDim;
        this.activation = activation;

        this.inputs = new double[inputDim][];

        this.kernel = this.initializeKernel();
        this.kernelDifference = new double[units][inputDim];
        this.useBias = useBias;
        this.bias = useBias ? this.initializeBias() : null;
        this.biasDifference = useBias ? new double[units] : null;

        this.signalError = new double[units][];
    }

    private double[][] initializeKernel() {
        var kernel = new double[units][inputDim];
        Random random = new Random();
        for (int outerIndex = 0; outerIndex < units; outerIndex++) {
            for (int innerIndex = 0; innerIndex < inputDim; innerIndex++) {
                kernel[outerIndex][innerIndex] = random.nextDouble() * 2.0 - 1.0;
            }
        }
        return kernel;
    }

    private double[] initializeBias() {
        var bias = new double[this.units];
        Random random = new Random();
        for (int index = 0; index < this.units; index++) {
            bias[index] = random.nextDouble();
        }
        return bias;
    }

    public Activation activation() {
        return activation;
    }

    public double[][] computeOutputs() {
        var outputs = LinAlg.matrixMultiplication(kernel, inputs);
        if (useBias) {
            var factor = inputs[0].length;
            var bias = LinAlg.broadcast(LinAlg.reshape(this.bias, units, 1), 0, factor);
            outputs = LinAlg.matrixAddition(outputs, bias);
        }
        return LinAlg.transform(outputs, activation::value);
    }

}
