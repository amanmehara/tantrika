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

import nn.activations.Activation;

public class Layer {

    public final Activation activation;

    //Input from previous layer
    public double[] inputs;

    public final Node[] nodes;

    //Constructor
    public Layer(Activation activation, int NumberOfNodes, int NumberOfInputs) {
        this.activation = activation;

        nodes = new Node[NumberOfNodes];

        for (int i = 0; i < NumberOfNodes; i++)
            nodes[i] = new Node(NumberOfInputs);

        inputs = new double[NumberOfInputs];
    }

    public Node node(int index) {
        return this.nodes[index];
    }

    public Node[] nodes() {
        return this.nodes;
    }

    // Calculates output for all the nodes in the current layer (except inputs layer)
    public void computeOutput() {
        for (Node node : nodes) {
            double output = node.threshold;
            for (int j = 0; j < node.weight.length; j++)
                output += inputs[j] * node.weight[j];
            node.output = this.activation.value(output);
        }
    }

    public double[] outputVector() {
        double[] outputs;
        outputs = new double[nodes.length];
        for (int i = 0; i < nodes.length; i++)
            outputs[i] = nodes[i].output;
        return outputs;
    }
}
