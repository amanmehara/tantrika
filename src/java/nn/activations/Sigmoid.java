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

package nn.activations;

public class Sigmoid implements Activation {

    @Override
    public double value(double input) {
        return 1.0 / (1.0 + Math.exp(-1.0 * input));
    }

    @Override
    public double derivative(double input) {
        return this.value(input) * (1.0 - this.value(input));
    }

}
