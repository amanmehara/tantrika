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

import com.amanmehara.tantrika.io.CSVReader;
import com.amanmehara.tantrika.math.linalg.Matrix;
import com.amanmehara.tantrika.math.linalg.Vector;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Testing {


    public static void main(String[] Args) throws IOException {

        int[] numberOfNodes = {30, 21, 1};

        var samples = new CSVReader(Paths.get("testing_dataset")).read();
        Matrix inputs = inputs(samples);
        Matrix outputs = outputs(samples);

        Vector weights = new Vector(readWeights("weights1").toArray(Double[]::new));

        BackPropagationTest backPropagationTest = new BackPropagationTest(
                numberOfNodes,
                inputs,
                weights);

        var actualOutputs = backPropagationTest.test();

        PrintWriter out;
        try {

            var meanDeviation = 0.0;
            var meanSquareError = 0.0;

            out = new PrintWriter("delta");
            int correct = 0;
            for (int i = 0; i < actualOutputs.innerSize(); i++) {
                var delta = actualOutputs.get(0, i) - outputs.get(0, i);
                out.println(Math.abs(delta));
                System.out.print(String.format("delta(%04d): ", i));
                System.out.println(Math.abs(delta));
                if (Math.abs(delta) < 0.1) {
                    correct++;
                }

                meanDeviation += delta;
                meanSquareError += Math.pow(delta, 2.0);
            }
            out.close();
            double accuracy = correct / (double) outputs.innerSize();
            double percentageAccuracy = correct / (double) outputs.innerSize() * 100;

            System.out.println(accuracy);
            System.out.println(percentageAccuracy);

            meanDeviation /= (double) outputs.innerSize();
            meanSquareError /= (double) outputs.innerSize();

            System.out.println(meanDeviation);
            System.out.println(meanSquareError);


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static Matrix inputs(Matrix samples) {
        var inputs = new Double[samples.outerSize()][samples.innerSize() - 1];
        for (var outerIndex = 0;
             outerIndex < samples.outerSize();
             outerIndex++) {
            for (var innerIndex = 0;
                 innerIndex < samples.innerSize() - 1;
                 innerIndex++) {
                inputs[outerIndex][innerIndex] = samples.get(outerIndex, innerIndex);
            }
        }
        return new Matrix(inputs).transpose();
    }

    private static Matrix outputs(Matrix samples) {
        var outputs = new Double[samples.outerSize()][1];
        for (var outerIndex = 0;
             outerIndex < samples.outerSize();
             outerIndex++) {
            outputs[outerIndex][0] = samples.get(outerIndex, samples.innerSize() - 1);
        }
        return new Matrix(outputs).transpose();
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

}
