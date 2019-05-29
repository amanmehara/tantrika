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

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Training {
    public static void main(String[] Args) {
        BufferedReader bufferedReader;

        List<double[]> samples = new ArrayList<>();
        String sample;


        Scanner consoleScanner = new Scanner(System.in);

        try {
            bufferedReader = new BufferedReader(new FileReader("training_dataset"));
            while ((sample = bufferedReader.readLine()) != null) {
                samples.add(Arrays
                        .stream(sample.split(","))
                        .map(String::trim)
                        .mapToDouble(Double::parseDouble)
                        .toArray());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.print("Number of Layers: ");
        int layers = consoleScanner.nextInt();
        System.out.println();

        int[] numberOfNodes = new int[layers];

        for (int i = 0; i < numberOfNodes.length; i++) {
            System.out.print("Layer " + i + " : ");
            numberOfNodes[i] = consoleScanner.nextInt();
        }

        System.out.println();

        var inputSamples = new Double[samples.size()][samples.get(0).length - 1];
        var outputSamples = new Double[samples.size()][1];

        for (int i = 0; i < samples.size(); i++) {
            for (int j = 0; j < samples.get(i).length - 1; j++) {
                inputSamples[i][j] = samples.get(i)[j];
            }
            outputSamples[i][0] = samples.get(i)[samples.get(i).length - 1];
        }

        BackPropagationTrain backPropagationTrain = new BackPropagationTrain(
                numberOfNodes,
                new Matrix(inputSamples).transpose(),
                new Matrix(outputSamples).transpose(),
                0.01,
                0.01,
                0.01,
                1024);

        backPropagationTrain.trainNetwork();

        System.out.println(backPropagationTrain.getError());

    }
}
