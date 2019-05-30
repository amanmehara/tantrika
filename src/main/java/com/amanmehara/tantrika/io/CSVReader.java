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

package com.amanmehara.tantrika.io;

import com.amanmehara.tantrika.math.linalg.Matrix;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.Stream;

public class CSVReader {

    private final Path path;

    public CSVReader(Path path) {
        this.path = path;
    }

    public Matrix read() throws IOException {
        Function<String, double[]> parseLine = line -> Arrays
                .stream(line.split(","))
                .map(String::trim)
                .mapToDouble(Double::parseDouble)
                .toArray();

        try (Stream<String> lines = Files.lines(path)) {
            return new Matrix(lines.map(parseLine).toArray(double[][]::new));
        }
    }

}
