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

    //Net input
    private double net;

    //Input from previous layer
    public double input[];

    // Nodes in current layer
    public Node node[];

    //Constructor
    public Layer(Activation activation, int NumberOfNodes, int NumberOfInputs) {
        this.activation = activation;

        node = new Node[NumberOfNodes];

        for (int i = 0; i < NumberOfNodes; i++)
            node[i] = new Node(NumberOfInputs);

        input = new double[NumberOfInputs];
    }

    // Getter
    public Node[] getNodes() {
        return node;
    }

    // Calculates output for all the nodes in the current layer (except input layer)
    public void computeOutput() {
        for (int i = 0; i < node.length; i++) {
            net = node[i].threshold;

            for (int j = 0; j < node[i].weight.length; j++)
                net += input[j] * node[i].weight[j];

            node[i].output = this.activation.value(net);
        }
    }

    // Return the vector containing output from all Nodes
    public double[] outputVector() {

        double vector[];

        vector = new double[node.length];

        for (int i = 0; i < node.length; i++)
            vector[i] = node[i].output;

        return (vector);
    }
}
