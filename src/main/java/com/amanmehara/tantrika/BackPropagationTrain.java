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

package com.amanmehara.tantrika;

import com.amanmehara.tantrika.math.linalg.Matrix;
import com.amanmehara.tantrika.nn.Layer;
import com.amanmehara.tantrika.nn.activations.Identity;
import com.amanmehara.tantrika.nn.activations.Tanh;
import com.amanmehara.tantrika.nn.initializers.RandomUniform;
import com.amanmehara.tantrika.nn.initializers.Zeros;

public class BackPropagationTrain {

    private double learningRate;

    private double momentum;

    private int numberOfLayers;

    private Layer[] layer;

    private int numberOfSamples;

    private int sampleNumber;

    private Matrix inputs;
    private Matrix outputs;

    private Matrix actualOutput;


    private long maximumEpochs;

    private double overallError;

    private double minimumError;

    //Constructor
    public BackPropagationTrain(int[] numberOfNodes,
                                Matrix inputs,
                                Matrix outputs,
                                double learningRate,
                                double moment,
                                double minimumError,
                                long maximumEpochs) {

        // Initiate variables
        this.numberOfSamples = inputs.innerSize();
        this.minimumError = minimumError;
        this.learningRate = learningRate;
        this.momentum = moment;
        this.numberOfLayers = numberOfNodes.length;
        this.maximumEpochs = maximumEpochs;

        this.inputs = inputs;
        this.outputs = outputs;

        // Create network layers
        this.layer = new Layer[numberOfLayers];

        // Input layer initialised
        layer[0] = new Layer(numberOfNodes[0], numberOfNodes[0], new Identity(), new RandomUniform(-1.0, 1.0));

        // Layers other than inputs layer initialised
        for (int i = 1; i < numberOfLayers; i++) {
            layer[i] = new Layer(numberOfNodes[i], numberOfNodes[i - 1], new Tanh(), new RandomUniform(-1.0, 1.0), new Zeros());
        }

        actualOutput = new Matrix(numberOfSamples, layer[numberOfLayers - 1].units());

    }

    // Getter
    public double getError() {
        calculateOverallError();
        return overallError;
    }

    // Calculate the nodes activations
    private Matrix feedForward(Matrix inputs) {
        var index = 1;
        while (index < numberOfLayers - 1) {
            inputs = layer[index].computeOutputs(inputs);
            index++;
        }
        return layer[index].computeOutputs(inputs);
    }

    // TODO: Vectorize (+ implicitly fix)
    private void updateWeights(Matrix outputs) {
//        calculateSignalErrors(outputs);
//        backPropagateError();
    }

//    private void calculateSignalErrors(double[][] outputs1) {
//
//        int outputLayer = numberOfLayers - 1;
//
//        // Calculate signal error for output layer
//        var outputs = layer[outputLayer].computeOutputs();
//        for (int i = 0; i < layer[outputLayer].units; i++) {
//            layer[outputLayer].signalError[i]
//                    = (outputs[sampleNumber][i] - outputs[i])
//                    * layer[outputLayer].activation().derivative(outputs[i]);
//        }
//
//        // Calculate signal error for rest of the layers
//        for (int i = numberOfLayers - 2; i > 0; i--) {
//            outputs = layer[i].computeOutputs();
//            for (int j = 0; j < layer[i].units; j++) {
//                double sum = 0;
//                for (int k = 0; k < layer[i + 1].units; k++) {
//                    sum += layer[i + 1].kernel[j][k] * layer[i + 1].signalError[k];
//                }
//                layer[i].signalError[j] = layer[i].activation().derivative(outputs[j]) * sum;
//            }
//        }
//    }
//
//    //Back-Propagation of error
//    private void backPropagateError() {
//
//        // Update Weights
//        for (int i = numberOfLayers - 1; i > 0; i--) {
//            for (int j = 0; j < layer[i].units; j++) {
//
//                // Calculate bias weight difference
//                layer[i].biasDifference[j]
//                        = (learningRate * layer[i].signalError[j])
//                        + (momentum * layer[i].biasDifference[j]);
//
//                // Update bias weight
//                layer[i].bias[j] += layer[i].biasDifference[j];
//
//                // Update weights
//                for (int k = 0; k < layer[i].inputs.length; k++) {
//                    // Calculate weight difference
//                    // outputs(layer(i-1)) = inputs(layer(i))
//                    layer[i].kernelDifference[k][j]
//                            = (learningRate * layer[i].signalError[j] * layer[i].inputs[k])
//                            + (momentum * layer[i].kernelDifference[k][j]);
//
//                    // Update weight
//                    layer[i].kernel[k][j] += layer[i].kernelDifference[k][j];
//                }
//            }
//        }
//    }

    //Compute overall error
    private void calculateOverallError() {
        overallError = 0;
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[numberOfLayers - 1].units(); j++) {
                overallError += (Math.pow(outputs.get(i, j) - actualOutput.get(i, j), 2));
            }
        }
        overallError /= numberOfSamples;
    }

    // com.amanmehara.tantrika.Training the Neural Network
    public void trainNetwork() {

        long epochs = 0;
        do {

//            actualOutput = feedForward(new Matrix(inputs).transpose()).transpose();
            updateWeights(actualOutput);
            epochs++;

            // Calculate Error Function
            calculateOverallError();
            System.out.println("Epoch " + epochs + ": " + this.getError());

        } while ((overallError > minimumError) && (epochs < maximumEpochs));
    }
}
