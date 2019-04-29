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

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Testing {
    private static double[][] inputSamples;
    static double[][] outputSamples;
    static double[] weightsArray;

    public static void main(String Args[]) {

        List<int[]> samples = readSampleData("testing_dataset");

        List<Double> weights = readWeights("weights1");


        int[] numberOfNodes = {30, 21, 1};

        initializeIOSamples(samples);

        initializeWeights(weights);

        BackPropagationTest backPropagationTest = new BackPropagationTest(numberOfNodes, inputSamples, outputSamples, weightsArray, 0.01, 0.01);

        backPropagationTest.testNetwork();

        PrintWriter out;
        try {

            double meanDeviation = 0;
            double meanSquareError = 0;

            out = new PrintWriter("delta");
            int correct = 0;
            for (int i = 0; i < backPropagationTest.actualOutput.length; i++) {
                double delta = backPropagationTest.actualOutput[i][0] - outputSamples[i][0];
                out.println(Math.abs(delta));
                System.out.println(Math.abs(delta));
                if (Math.abs(delta) < 1) {
                    correct++;
                }

                meanDeviation += delta;
                meanSquareError += Math.pow(delta, 2);
            }
            out.close();
            double accuracy = correct / (double) outputSamples.length;
            double percentageAccuracy = correct / (double) outputSamples.length * 100;

            System.out.println(accuracy);
            System.out.println(percentageAccuracy);

            meanDeviation /= (double) outputSamples.length;
            meanSquareError /= (double) outputSamples.length;

            System.out.println(meanDeviation);
            System.out.println(meanSquareError);


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static void initializeWeights(List<Double> weights) {
        weightsArray = new double[weights.size()];

        for (int i = 0; i < weights.size(); i++) {
            weightsArray[i] = weights.get(i);
        }
    }

    private static void initializeIOSamples(List<int[]> samples) {
        inputSamples = new double[samples.size()][samples.get(0).length - 1];
        outputSamples = new double[samples.size()][1];

        for (int i = 0; i < samples.size(); i++) {
            for (int j = 0; j < samples.get(i).length - 1; j++) {
                inputSamples[i][j] = samples.get(i)[j];
            }
            outputSamples[i][0] = samples.get(i)[samples.get(i).length - 1];
        }
    }

    private static List<Double> readWeights(String fileName) {
        BufferedReader bufferedReader;

        List<Double> weights = new ArrayList<>();
        String weight;

        try {
            bufferedReader = new BufferedReader(new FileReader(fileName));
            while ((weight = bufferedReader.readLine()) != null) {
                weights.add(Double.parseDouble(weight));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return weights;
    }

    private static List<int[]> readSampleData(String fileName) {
        BufferedReader bufferedReader;

        List<int[]> samples = new ArrayList<>();
        String sample;

        try {
            bufferedReader = new BufferedReader(new FileReader(fileName));
            while ((sample = bufferedReader.readLine()) != null) {
                samples.add(Arrays
                        .stream(sample.split(","))
                        .map(String::trim)
                        .mapToInt(Integer::parseInt)
                        .toArray());
            }
            bufferedReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return samples;
    }
}
