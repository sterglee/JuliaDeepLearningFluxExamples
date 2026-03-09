public class MLP {
    public static void main(String[] args) {
        int N = 1000, in = 2, hid = 4;
        double[][] X = new double[N][in];
        double[] y = new double[N];

        for(int i=0; i<N; i++) {
            X[i][0] = Math.random();
            X[i][1] = Math.random();
            y[i] = (X[i][0]*X[i][0] + X[i][1]*X[i][1] > 0.5) ? 1.0 : 0.0;
        }

        double[][] W1 = new double[hid][in];
        double[] b1 = new double[hid];
        double[] W2 = new double[hid];
        double b2 = 0;

        long start = System.nanoTime();

        int epochs=1000000;
        for(int epoch=0; epoch<epochs; epoch++) {
            for(int i=0; i<N; i++) {
                // Forward
                double[] h = new double[hid];
                for(int j=0; j<hid; j++) {
                    double z = b1[j];
                    for(int k=0; k<in; k++) z += W1[j][k] * X[i][k];
                    h[j] = Math.max(0, z);
                }
                double z2 = b2;
                for(int j=0; j<hid; j++) z2 += W2[j] * h[j];
                double pred = 1.0 / (1.0 + Math.exp(-z2));

                // Backprop
                double dz2 = pred - y[i];
                for(int j=0; j<hid; j++) {
                    double dW2 = dz2 * h[j];
                    double dh = (h[j] > 0) ? (dz2 * W2[j]) : 0;
                    W2[j] -= 0.01 * dW2;
                    for(int k=0; k<in; k++) W1[j][k] -= 0.01 * dh * X[i][k];
                    b1[j] -= 0.01 * dh;
                }
                b2 -= 0.01 * dz2;
            }
        }

        long end = System.nanoTime();
        System.out.println("Java Training Time: " + (end - start)/1e9 + "s");
    }
}
