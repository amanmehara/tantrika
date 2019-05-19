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
import nn.activations.Activation;
import nn.optimizers.GradientDescent;
import nn.optimizers.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class Model {

    private final List<Layer> layers;
    private final Optimizer<Model> optimizer;

    private Model(List<Layer> layers, Optimizer<Model> optimizer) {
        this.layers = layers;
        this.optimizer = optimizer;
    }

    public List<Layer> layers() {
        return layers;
    }

    public Matrix feedForward(Matrix inputs) {
        return null;
    }

    public void train(Matrix inputs, double[] outputs, int epochs) {
        IntStream.range(0, epochs).forEach(epoch -> {
            feedForward(inputs);
            optimizer.optimize(this);
        });

    }

    public Matrix test(Matrix inputs, double[] outputs) {
        return feedForward(inputs);
    }

    public static class Builder {

        private List<Layer> layers;
        private Optimizer<Model> optimizer;

        public Builder() {
            this.layers = new ArrayList<>();
            this.optimizer = new GradientDescent<>();
        }

        public Builder addLayer(Layer layer) {
            layers.add(layer);
            return this;
        }

        public Builder addLayer(int units, int inputDim, Activation activation, boolean useBias) {
            if (layers.size() > 0) {
                throw new IllegalStateException();
            }

            layers.add(new Layer(units, inputDim, activation, useBias));
            return this;
        }

        public Builder addLayer(int units, Activation activation, boolean useBias) {
            if (layers.size() == 0) {
                throw new IllegalStateException();
            }

            int inputDim = layers.get(layers.size() - 1).units();
            layers.add(new Layer(units, inputDim, activation, useBias));
            return this;

        }

        public Builder optimizer(Optimizer<Model> optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        public Model build() {
            return new Model(layers, optimizer);
        }

    }

}
