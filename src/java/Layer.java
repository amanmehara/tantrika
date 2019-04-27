public class Layer {

    //Net input
    private double net;

    //Input from previous layer
    public	double input[];

    // Nodes in current layer
    public Node node[];

    //Constructor
    public Layer (int NumberOfNodes, int NumberOfInputs) {
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

            node[i].output = activationFunctionTanH(net);
        }
    }

    // Activation function (Sigmoid)
    private double activationFunctionSigmoid(double Net) {
        return 1/(1+Math.exp(-Net));
    }

    // Activation function (TanH)
    private double activationFunctionTanH(double Net) {
        return Math.tanh(Net);
    }

    // Return the vector containing output from all Nodes
    public double[] outputVector() {

        double vector[];

        vector = new double[node.length];

        for (int i=0; i < node.length; i++)
            vector[i] = node[i].output;

        return (vector);
    }
}
