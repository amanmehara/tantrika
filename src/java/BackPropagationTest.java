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

import java.util.stream.IntStream;

public class BackPropagationTest {

    private int numberOfLayers;

    private Layer[] layer;

    private int numberOfSamples;

    private double[][] input;

    public double[][] actualOutput;

    private double[][] desiredOutput;

    //Constructor
    public BackPropagationTest(int[] numberOfNodes,
                               double[][] inputSamples,
                               double[][] outputSamples,
                               double[] weights) {

        // Initiate variables
        this.numberOfSamples = inputSamples.length;
        this.numberOfLayers = numberOfNodes.length;

        // Create network layers
        this.layer = new Layer[numberOfLayers];

        // Input layer initialised
        layer[0] = new Layer(numberOfNodes[0], numberOfNodes[0], new Identity(), false);

        // Layers other than input layer initialised
        for (int i = 1; i < numberOfLayers; i++) {
            layer[i] = new Layer(numberOfNodes[i], numberOfNodes[i - 1], new Tanh(), true);
        }

        input = new double[numberOfSamples][layer[0].units];

        desiredOutput = new double[numberOfSamples][layer[numberOfLayers - 1].units];

        actualOutput = new double[numberOfSamples][layer[numberOfLayers - 1].units];

        // Assign Input Set
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[0].units; j++) {
                input[i][j] = inputSamples[i][j];
            }
        }

        // Assign Output Set
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[numberOfLayers - 1].units; j++) {
                desiredOutput[i][j] = outputSamples[i][j];
            }
        }

        // Assign Weights
        int weightsCount = 0;
        for (int i = 0; i < layer.length; i++) {
            for (int j = 0; j < layer[i].units; j++) {
                for (int k = 0; k < layer[i].kernel.length; k++) {
                    layer[i].kernel[k][j] = weights[weightsCount];
                    weightsCount++;
                }
            }
        }
    }

    // Calculate the nodes activations
    private double[] feedForward() {

        int i;

//        for (i = 0; i < layer[0].units; i++) {
//            layer[0].nodes[i].output = layer[0].inputs[i];
//        }

        layer[1].inputs = layer[0].inputs;
        for (i = 1; i < numberOfLayers; i++) {
            var outputs = layer[i].computeOutputs();
            if (i != numberOfLayers - 1)
                layer[i + 1].inputs = outputs;
            else
                return outputs;
        }

        // Making the compiler happy.
        return null;
    }

    // Test the Neural Network
    public void testNetwork() {

        IntStream.range(0, numberOfSamples).forEach(sampleNumber -> {
            for (int i = 0; i < layer[0].units; i++) {
                layer[0].inputs[i] = input[sampleNumber][i];
            }
            var outputs = this.feedForward();

            // Assign actualOutput
            for (int i = 0; i < layer[numberOfLayers - 1].units; i++) {
                actualOutput[sampleNumber][i] = outputs[i];
            }
        });
    }
}
