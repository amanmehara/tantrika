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

public class Node {

    public double output;

    public double[] weight;

    public double threshold;

    public double[] weightDiff;

    public double thresholdDiff;

    public double signalError;

    //Constructor
    public Node(int numberOfNodes) {
        weight = new double[numberOfNodes];
        weightDiff = new double[numberOfNodes];
        initialiseWeights();
    }

    //Getter
    public double[] getWeights() {
        return weight;
    }

    //Initialise weights & threshold
    private void initialiseWeights() {

        threshold = -1 + 2 * Math.random();
        thresholdDiff = 0;

        for (int i = 0; i < weight.length; i++) {
            weight[i] = -1 + 2 * Math.random();
            weightDiff[i] = 0;
        }

    }

}
