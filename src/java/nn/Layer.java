package nn;/*
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

import math.LinAlg;
import nn.activations.Activation;

import java.util.Arrays;
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

    public double[] signalError;

    public double[] inputs;

    //Constructor
    public Layer(int units, int inputDim, Activation activation, boolean useBias) {
        this.units = units;
        this.inputDim = inputDim;
        this.activation = activation;

        this.inputs = new double[inputDim];

        this.kernel = this.initializeKernel();
        this.kernelDifference =  new double[inputDim][units];
        this.useBias = useBias;
        this.bias = useBias ? this.initializeBias() : null;
        this.biasDifference = useBias ? new double[units]: null;

        this.signalError = new double[units];
    }

    private double[][] initializeKernel() {
        var kernel = new double[this.inputDim][this.units];
        Random random = new Random();
        for (int outerIndex = 0; outerIndex < this.inputDim; outerIndex++) {
            for (int innerIndex = 0; innerIndex < this.units; innerIndex++) {
                kernel[outerIndex][innerIndex] = random.nextDouble() * 2.0 - 1.0;
            }
        }
        return kernel;
    }

    private double[] initializeBias() {
        var bias = new double[this.units];
        Random random = new Random();
        for (int index = 0; index < this.units; index++) {
            bias[index] =  random.nextDouble();
        }
        return bias;
    }

    public Activation activation() {
        return this.activation;
    }

    public double[] computeOutputs() {
        var inputs = LinAlg.reshape(this.inputs, 1, this.inputDim);
        var outputs = LinAlg.matrixMultiplication(inputs, this.kernel);
        if (this.useBias) {
            var bias = LinAlg.reshape(this.bias, 1, this.units);
            outputs = LinAlg.matrixAddition(outputs, bias);
        }
        return Arrays.stream(LinAlg.reshape(outputs)).map(this.activation::value).toArray();
    }

}
