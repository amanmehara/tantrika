import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Learning {
    public static void main(String Args[]) {
        BufferedReader bufferedReader;

        List<int[]> samples = new ArrayList<>();
        String sample;

        try {
            bufferedReader=new BufferedReader(new FileReader("training_dataset"));
            while ((sample=bufferedReader.readLine())!=null) {

                samples.add(Arrays
                        .stream(sample.split(","))
                        .map(String::trim)
                        .mapToInt(Integer::parseInt)
                        .toArray());

                //samples.add(sample.split(","));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int[] args: samples) {
            int sum=0;
            for (int arg: args) {
                sum+=arg;
            }
            System.out.println(sum);
        }
    }
}