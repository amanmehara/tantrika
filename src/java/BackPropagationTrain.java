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

import nn.activations.Identity;
import nn.activations.Tanh;

public class BackPropagationTrain {

    private double learningRate;

    private double momentum;

    private int numberOfLayers;

    private Layer[] layer;

    private int numberOfSamples;

    private int sampleNumber;

    private double[][] input;

    private double[][] actualOutput;

    private double[][] desiredOutput;

    private long maximumEpochs;

    private double overallError;

    private double minimumError;

    //Constructor
    public BackPropagationTrain(int[] numberOfNodes,
                                double[][] inputSamples,
                                double[][] outputSamples,
                                double learningRate,
                                double moment,
                                double minimumError,
                                long maximumEpochs) {

        // Initiate variables
        this.numberOfSamples = inputSamples.length;
        this.minimumError = minimumError;
        this.learningRate = learningRate;
        this.momentum = moment;
        this.numberOfLayers = numberOfNodes.length;
        this.maximumEpochs = maximumEpochs;

        // Create network layers
        this.layer = new Layer[numberOfLayers];

        // Input layer initialised
        layer[0] = new Layer(new Identity(), numberOfNodes[0], numberOfNodes[0]);

        // Layers other than inputs layer initialised
        for (int i = 1; i < numberOfLayers; i++) {
            layer[i] = new Layer(new Tanh(), numberOfNodes[i], numberOfNodes[i - 1]);
        }

        input = new double[numberOfSamples][layer[0].nodes.length];

        desiredOutput = new double[numberOfSamples][layer[numberOfLayers - 1].nodes.length];

        actualOutput = new double[numberOfSamples][layer[numberOfLayers - 1].nodes.length];

        // Assign Input Set
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[0].nodes.length; j++) {
                input[i][j] = inputSamples[i][j];
            }
        }

        // Assign Output Set
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[numberOfLayers - 1].nodes.length; j++) {
                desiredOutput[i][j] = outputSamples[i][j];
            }
        }
    }

    // Getter
    public double getError() {
        calculateOverallError();
        return overallError;
    }

    // Calculate the nodes activations
    public void feedForward() {

        int i, j;

        for (i = 0; i < layer[0].nodes.length; i++) {
            layer[0].nodes[i].output = layer[0].inputs[i];
        }

        layer[1].inputs = layer[0].inputs;
        for (i = 1; i < numberOfLayers; i++) {
            layer[i].computeOutput();
            if (i != numberOfLayers - 1)
                layer[i + 1].inputs = layer[i].outputVector();
        }

    }

    private void updateWeights() {
        calculateSignalErrors();
        backPropagateError();
    }

    private void calculateSignalErrors() {

        int outputLayer = numberOfLayers - 1;

        // Calculate signal error for output layer
        for (int i = 0; i < layer[outputLayer].nodes.length; i++) {
            layer[outputLayer].nodes[i].signalError
                    = (desiredOutput[sampleNumber][i] - layer[outputLayer].nodes[i].output)
                    * layer[outputLayer].activation.derivative(layer[outputLayer].nodes[i].output);
        }

        // Calculate signal error for rest of the layers
        for (int i = numberOfLayers - 2; i > 0; i--) {
            for (int j = 0; j < layer[i].nodes.length; j++) {
                double sum = 0;
                for (int k = 0; k < layer[i + 1].nodes.length; k++) {
                    sum += layer[i + 1].nodes[k].weight[j] * layer[i + 1].nodes[k].signalError;
                }
                layer[i].nodes[j].signalError = layer[i].activation.derivative(layer[i].nodes[j].output) * sum;
            }
        }
    }

    //Back-Propagation of error
    private void backPropagateError() {

        // Update Weights
        for (int i = numberOfLayers - 1; i > 0; i--) {
            for (int j = 0; j < layer[i].nodes.length; j++) {

                // Calculate bias weight difference
                layer[i].nodes[j].thresholdDiff
                        = (learningRate * layer[i].nodes[j].signalError)
                        + (momentum * layer[i].nodes[j].thresholdDiff);

                // Update bias weight
                layer[i].nodes[j].threshold = layer[i].nodes[j].threshold + layer[i].nodes[j].thresholdDiff;

                // Update weights
                for (int k = 0; k < layer[i].inputs.length; k++) {
                    // Calculate weight difference
                    layer[i].nodes[j].weightDiff[k]
                            = (learningRate * layer[i].nodes[j].signalError * layer[i - 1].nodes[k].output)
                            + (momentum * layer[i].nodes[j].weightDiff[k]);

                    // Update weight
                    layer[i].nodes[j].weight[k] = layer[i].nodes[j].weight[k] + layer[i].nodes[j].weightDiff[k];
                }
            }
        }
    }

    //Compute overall error
    private void calculateOverallError() {
        overallError = 0;
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[numberOfLayers - 1].nodes.length; j++) {
                overallError += (Math.pow(desiredOutput[i][j] - actualOutput[i][j], 2));
            }
        }
        overallError /= numberOfSamples;
    }

    // Training the Neural Network
    public void trainNetwork() {

        long epochs = 0;
        do {
            for (sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {

                for (int i = 0; i < layer[0].nodes.length; i++) {
                    layer[0].inputs[i] = input[sampleNumber][i];
                }
                this.feedForward();

                // Assign actualOutput
                for (int i = 0; i < layer[numberOfLayers - 1].nodes.length; i++) {
                    actualOutput[sampleNumber][i] = layer[numberOfLayers - 1].nodes[i].output;
                }
                this.updateWeights();
            }

            epochs++;

//            if(epochs>2000) {
//                this.momentum=0.9;
//            }

            // Calculate Error Function
            calculateOverallError();
            System.out.println("Epoch " + epochs + ": " + this.getError());
        } while ((overallError > minimumError) && (epochs < maximumEpochs));
    }
}
