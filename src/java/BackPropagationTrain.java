import nn.activations.Identity;
import nn.activations.Tanh;

public class BackPropagationTrain {

    private double learningRate;

    private double momentum;

    private int numberOfLayers;

    public Layer layer[];

    private int numberOfSamples;

    private int sampleNumber;

    private double input[][];

    public double actualOutput[][];

    private double desiredOutput[][];

    private long maximumEpochs;

    private double overallError;

    private double minimumError;

    //Constructor
    public BackPropagationTrain(int numberOfNodes[],
                                double inputSamples[][],
                                double outputSamples[][],
                                double learningRate,
                                double moment,
                                double minimumError,
                                long maximumEpochs) {

        // Initiate variables
        this.numberOfSamples = inputSamples.length;
        this.minimumError = minimumError;
        this.learningRate = learningRate;
        this.momentum = moment;
        this.numberOfLayers = numberOfNodes.length;
        this.maximumEpochs = maximumEpochs;

        // Create network layers
        this.layer = new Layer[numberOfLayers];

        // Input layer initialised
        layer[0] = new Layer(new Identity(), numberOfNodes[0], numberOfNodes[0]);

        // Layers other than input layer initialised
        for (int i = 1; i < numberOfLayers; i++) {
            layer[i] = new Layer(new Tanh(), numberOfNodes[i], numberOfNodes[i - 1]);
        }

        input = new double[numberOfSamples][layer[0].node.length];

        desiredOutput = new double[numberOfSamples][layer[numberOfLayers - 1].node.length];

        actualOutput = new double[numberOfSamples][layer[numberOfLayers - 1].node.length];

        // Assign Input Set
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[0].node.length; j++) {
                input[i][j] = inputSamples[i][j];
            }
        }

        // Assign Output Set
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[numberOfLayers - 1].node.length; j++) {
                desiredOutput[i][j] = outputSamples[i][j];
            }
        }
    }

    //Getter
    public Layer[] getLayers() {
        return layer;
    }

    // Getter
    public double getError() {
        calculateOverallError();
        return overallError;
    }

    // Calculate the node activations
    public void feedForward() {

        int i, j;

        for (i = 0; i < layer[0].node.length; i++) {
            layer[0].node[i].output = layer[0].input[i];
        }

        layer[1].input = layer[0].input;
        for (i = 1; i < numberOfLayers; i++) {
            layer[i].computeOutput();
            if (i != numberOfLayers - 1)
                layer[i + 1].input = layer[i].outputVector();
        }

    }

    public void updateWeights() {
        calculateSignalErrors();
        backPropagateError();
    }

    private void calculateSignalErrors() {

        int outputLayer = numberOfLayers - 1;

        // Calculate signal error for output layer
        for (int i = 0; i < layer[outputLayer].node.length; i++) {
            layer[outputLayer].node[i].signalError
                    = (desiredOutput[sampleNumber][i] - layer[outputLayer].node[i].output)
                    * layer[outputLayer].activation.derivative(layer[outputLayer].node[i].output);
        }

        // Calculate signal error for rest of the layers
        for (int i = numberOfLayers - 2; i > 0; i--) {
            for (int j = 0; j < layer[i].node.length; j++) {
                double sum = 0;
                for (int k = 0; k < layer[i + 1].node.length; k++) {
                    sum += layer[i + 1].node[k].weight[j] * layer[i + 1].node[k].signalError;
                }
                layer[i].node[j].signalError = layer[i].activation.derivative(layer[i].node[j].output) * sum;
            }
        }
    }

    //Back-Propagation of error
    private void backPropagateError() {

        // Update Weights
        for (int i = numberOfLayers - 1; i > 0; i--) {
            for (int j = 0; j < layer[i].node.length; j++) {

                // Calculate bias weight difference
                layer[i].node[j].thresholdDiff
                        = (learningRate * layer[i].node[j].signalError)
                        + (momentum * layer[i].node[j].thresholdDiff);

                // Update bias weight
                layer[i].node[j].threshold = layer[i].node[j].threshold + layer[i].node[j].thresholdDiff;

                // Update weights
                for (int k = 0; k < layer[i].input.length; k++) {
                    // Calculate weight difference
                    layer[i].node[j].weightDiff[k]
                            = (learningRate * layer[i].node[j].signalError * layer[i - 1].node[k].output)
                            + (momentum * layer[i].node[j].weightDiff[k]);

                    // Update weight
                    layer[i].node[j].weight[k] = layer[i].node[j].weight[k] + layer[i].node[j].weightDiff[k];
                }
            }
        }
    }

    //Compute overall error
    private void calculateOverallError() {
        overallError = 0;
        for (int i = 0; i < numberOfSamples; i++) {
            for (int j = 0; j < layer[numberOfLayers - 1].node.length; j++) {
                overallError += (Math.pow(desiredOutput[i][j] - actualOutput[i][j], 2));
            }
        }
        overallError /= numberOfSamples;
    }

    // Training the Neural Network
    public void trainNetwork() {

        long epochs = 0;
        do {
            for (sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {

                for (int i = 0; i < layer[0].node.length; i++) {
                    layer[0].input[i] = input[sampleNumber][i];
                }
                this.feedForward();

                // Assign actualOutput
                for (int i = 0; i < layer[numberOfLayers - 1].node.length; i++) {
                    actualOutput[sampleNumber][i] = layer[numberOfLayers - 1].node[i].output;
                }
                this.updateWeights();
            }

            epochs++;

//            if(epochs>2000) {
//                this.momentum=0.9;
//            }

            // Calculate Error Function
            calculateOverallError();
            System.out.println("Epoch " + epochs + ": " + this.getError());
        } while ((overallError > minimumError) && (epochs < maximumEpochs));
    }
}
