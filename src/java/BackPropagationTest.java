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

import math.linalg.Matrix;
import nn.Layer;
import nn.activations.Identity;
import nn.activations.Tanh;
import nn.initializers.RandomUniform;
import nn.initializers.Zeros;

public class BackPropagationTest {

    private int numberOfLayers;
    private Layer[] layers;
    private double[][] inputs;

    public BackPropagationTest(int[] numberOfNodes, double[][] inputSamples, double[] weights) {

        // Initiate variables
        int numberOfSamples = inputSamples.length;
        this.numberOfLayers = numberOfNodes.length;

        layers = new Layer[numberOfLayers];
        layers[0] = new Layer(numberOfNodes[0], numberOfNodes[0], new Identity(), new RandomUniform(-1.0, 1.0));
        for (int i = 1; i < numberOfLayers; i++) {
            layers[i] = new Layer(numberOfNodes[i], numberOfNodes[i - 1], new Tanh(), new RandomUniform(-1.0, 1.0), new Zeros());
        }

        inputs = new double[numberOfSamples][layers[0].units()];

        // Assign Input Set
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layers[0].units(); j++) {
                inputs[i][j] = inputSamples[i][j];
            }
        }

        // TODO: Enhance serialization and deserialization process.
        int weightsCount = 0;
        for (Layer layer : layers) {
            var kernel = new double[layer.units()][layer.inputDimension()];
            for (int k = 0; k < layer.units(); k++) {
                for (int j = 0; j < layer.inputDimension(); j++) {
                    kernel[k][j] = weights[weightsCount];
                    weightsCount++;
                }
            }
            layer.kernel(new Matrix(kernel));
        }
    }

    private Matrix feedForward(Matrix inputs) {
        var index = 1;
        while (index < numberOfLayers - 1) {
            inputs = layers[index].computeOutputs(inputs);
            index++;
        }
        return layers[index].computeOutputs(inputs);
    }

    public Matrix test() {
        return feedForward(new Matrix(inputs).transpose()).transpose();
    }

}
