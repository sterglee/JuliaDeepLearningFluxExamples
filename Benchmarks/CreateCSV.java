
import java.io.*;
import java.util.Random;

public class CreateCSV {
    public static void main(String[] args) {
        String filename = "large_data_java.csv";
        int rows = 25_000_000;
        Random rand = new Random();

        // Έναρξη χρονομέτρησης
        long startTime = System.nanoTime();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("ID,Value1,Value2,Category\n");

            for (int i = 1; i <= rows; i++) {
                // Χρήση StringBuilder για καλύτερη απόδοση στην ένωση String
                writer.write(i + "," + rand.nextDouble() + "," + rand.nextInt() + ",ABCDE\n");

                if (i % 5_000_000 == 0) {
                    System.out.println("Πρόοδος: " + i + " γραμμές...");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Λήξη χρονομέτρησης
        long endTime = System.nanoTime();
        double durationSeconds = (endTime - startTime) / 1_000_000_000.0;

        System.out.println("------------------------------------");
        System.out.println("Συνολικός χρόνος: " + String.format("%.2f", durationSeconds) + " δευτερόλεπτα");
        System.out.println("Ταχύτητα: " + String.format("%.0f", rows / durationSeconds) + " γραμμές/δευτ.");
    }
}

