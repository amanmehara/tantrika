package com.amanmehara.tantrika;/*
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

import com.amanmehara.tantrika.math.linalg.Matrix;
import com.amanmehara.tantrika.math.linalg.Vector;
import com.amanmehara.tantrika.nn.Layer;
import com.amanmehara.tantrika.nn.activations.Identity;
import com.amanmehara.tantrika.nn.activations.Tanh;
import com.amanmehara.tantrika.nn.initializers.RandomUniform;
import com.amanmehara.tantrika.nn.initializers.Zeros;

import java.util.ArrayList;
import java.util.List;

public class BackPropagationTest {

    private int numberOfLayers;
    private List<Layer> layers;
    private Matrix inputs;

    public BackPropagationTest(int[] numberOfNodes,
                               Matrix inputs,
                               Vector weights) {

        this.inputs = inputs;
        this.numberOfLayers = numberOfNodes.length;

        layers = new ArrayList<>();
        layers.add(new Layer(numberOfNodes[0],
                numberOfNodes[0],
                new Identity(),
                new RandomUniform(-1.0, 1.0)));
        for (int i = 1; i < numberOfLayers; i++) {
            layers.add(new Layer(
                    numberOfNodes[i],
                    numberOfNodes[i - 1],
                    new Tanh(),
                    new RandomUniform(-1.0, 1.0),
                    new Zeros()));
        }

        // TODO: Enhance serialization and deserialization process.
        int weightsCount = 0;
        for (Layer layer : layers) {
            var kernel = new Double[layer.units()][layer.inputDimension()];
            for (int k = 0; k < layer.units(); k++) {
                for (int j = 0; j < layer.inputDimension(); j++) {
                    kernel[k][j] = weights.get(weightsCount);
                    weightsCount++;
                }
            }
            layer.kernel(new Matrix(kernel));

            if (layer.useBias()) {
                // TODO: Persist bias.
                layer.bias(new Zeros().initializeVector(layer.units()));
            }
        }
    }

    private Matrix feedForward(Matrix inputs) {
        var index = 1;
        while (index < numberOfLayers - 1) {
            inputs = layers.get(index).computeOutputs(inputs);
            index++;
        }
        return layers.get(index).computeOutputs(inputs);
    }

    public Matrix test() {
        return feedForward(inputs);
    }

}
