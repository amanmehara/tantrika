public class Node {

    public	double 	output;

    public	double 	weight[];

    public	double	threshold;

    public	double	weightDiff[];

    public	double	thresholdDiff;

    public	double	signalError;

    //Constructor
    public Node (int numberOfNodes) {
        weight = new double[numberOfNodes];
        weightDiff = new double[numberOfNodes];
        initialiseWeights();
    }

    //Getter
    public double[] getWeights() {
        return weight;
    }

    //Getter
    public double getOutput() {
        return output;
    }

    //Initialise weights & threshold
    private void initialiseWeights() {

        threshold = -1+2*Math.random();
        thresholdDiff = 0;

        for(int i = 0; i < weight.length; i++) {
            weight[i]= -1+2*Math.random();
            weightDiff[i] = 0;
        }

    }

};
