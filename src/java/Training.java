import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Training {
    public static void main(String Args[]) {
        BufferedReader bufferedReader;

        List<int[]> samples = new ArrayList<>();
        String sample;


        Scanner consoleScanner = new Scanner(System.in);

        try {
            bufferedReader=new BufferedReader(new FileReader("training_dataset"));
            while ((sample=bufferedReader.readLine())!=null) {
                samples.add(Arrays
                        .stream(sample.split(","))
                        .map(String::trim)
                        .mapToInt(Integer::parseInt)
                        .toArray());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.print("Number of Layers: ");
        int layers = consoleScanner.nextInt();
        System.out.println();

        int[] numberOfNodes=new int[layers];

        for (int i=0; i<numberOfNodes.length; i++) {
            System.out.print("Layer " + i + " : ");
            numberOfNodes[i] = consoleScanner.nextInt();
        }

        System.out.println();

        double[][] inputSamples=new double[samples.size()][samples.get(0).length-1];
        double[][] outputSamples=new double[samples.size()][1];

        for (int i=0; i<samples.size(); i++) {
            for(int j=0;j<samples.get(i).length-1;j++) {
                inputSamples[i][j]=samples.get(i)[j];
            }
//            outputSamples[i][0]=(samples.get(i)[samples.get(i).length-1]==1?1:0);
            outputSamples[i][0]=samples.get(i)[samples.get(i).length-1];
        }

        //Initializing the Back Propagation Neural Network
        BackPropagationTrain backPropagationTrain = new BackPropagationTrain(numberOfNodes, inputSamples, outputSamples, 0.01, 0.01, 0.01, 1024);

        backPropagationTrain.trainNetwork();

        PrintWriter out = null;
        try {
            out = new PrintWriter("actual_output");

            for(int i = 0; i< backPropagationTrain.actualOutput.length; i++) {
                System.out.println(i + " : " + outputSamples[i][0] + " : " + backPropagationTrain.actualOutput[i][0]);
                out.println(backPropagationTrain.actualOutput[i][0]);
            }

            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        PrintWriter weightsWriter;
        try {
            weightsWriter=new PrintWriter("weights1");
            for(int i = 0; i< backPropagationTrain.layer.length; i++) {
                for(int j = 0; j< backPropagationTrain.layer[i].getNodes().length; j++) {
                    for (int k = 0; k< backPropagationTrain.layer[i].node[j].getWeights().length; k++) {
                        System.out.println("layer: " + i + ", node: " + j + " - weight "+ k +" || "+ backPropagationTrain.layer[i].node[j].weight[k]);
                        weightsWriter.println(backPropagationTrain.layer[i].node[j].weight[k]);
                    }
                }
            }
            weightsWriter.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        PrintWriter thresholdWriter;
        try {
            thresholdWriter=new PrintWriter("threshold");
            for(int i = 0; i< backPropagationTrain.layer.length; i++) {
                for(int j = 0; j< backPropagationTrain.layer[i].getNodes().length; j++) {
                        System.out.println("layer: " + i + ", node: " + j + " - bias " +" || "+ backPropagationTrain.layer[i].node[j].threshold);
                        thresholdWriter.println(backPropagationTrain.layer[i].node[j].threshold);
                }
            }
            thresholdWriter.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println(backPropagationTrain.getError());

    }
}
