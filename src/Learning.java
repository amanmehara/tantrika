import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class Learning {
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
            outputSamples[i][0]=samples.get(i)[samples.get(i).length-1];
        }

        //Initializing the Back Propagation Neural Network
        BackPropagation backPropagation = new BackPropagation(numberOfNodes, inputSamples, outputSamples, 0.01, 0.01, 0.0001, 1000);

        backPropagation.trainNetwork();

        for(int i=0;i<backPropagation.actualOutput.length;i++) {
           System.out.println(i + " : " + backPropagation.actualOutput[i][1]);
        }

    }
}