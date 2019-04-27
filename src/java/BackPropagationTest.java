public class BackPropagationTest {

    private double learningRate;

    private double momentum;

    private int numberOfLayers;

    public Layer layer[];

    private int numberOfSamples;

    private int sampleNumber;

    private double input[][];

    public double actualOutput[][];

    private double desiredOutput[][];

    private double weights[];

    //Constructor
    public BackPropagationTest(int numberOfNodes[],
                               double inputSamples[][],
                               double outputSamples[][],
                               double weights[],
                               double learningRate,
                               double momentum) {

        // Initiate variables
        this.numberOfSamples = inputSamples.length;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.numberOfLayers = numberOfNodes.length;

        // Create network layers
        this.layer = new Layer[numberOfLayers];

        // Input layer initialised
        layer[0] = new Layer(numberOfNodes[0], numberOfNodes[0]);

        // Layers other than input layer initialised
        for (int i = 1; i < numberOfLayers; i++) {
            layer[i] = new Layer(numberOfNodes[i], numberOfNodes[i - 1]);
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

        // Assign Weights
        int weightsCount = 0;
        for (int i = 0; i < layer.length; i++) {
            for (int j = 0; j < layer[i].node.length; j++) {
                for (int k = 0; k < layer[i].node[j].weight.length; k++) {
                    layer[i].node[j].weight[k] = weights[weightsCount];
                    weightsCount++;
                }
            }
        }
    }

    //Getter
    public Layer[] getLayers() {
        return layer;
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

    private double derivativeActivationFunctionTanH(double x) {
        return 1 - Math.pow(x, 2);
    }

    private double derivativeActivationFunctionSigmoid(double x) {
        return x * (1 - x);
    }

    // Test the Neural Network
    public void testNetwork() {

        for (sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {

            for (int i = 0; i < layer[0].node.length; i++) {
                layer[0].input[i] = input[sampleNumber][i];
            }
            this.feedForward();

            // Assign actualOutput
            for (int i = 0; i < layer[numberOfLayers - 1].node.length; i++) {
                actualOutput[sampleNumber][i] = layer[numberOfLayers - 1].node[i].output;
            }
        }
    }
}
