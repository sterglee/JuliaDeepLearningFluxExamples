
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

int main() {
    const int rows = 25000000;
    const string filename = "large_data_cpp.csv";

    // Προετοιμασία γεννήτριας τυχαίων αριθμών
    mt19937 gen(42); 
    uniform_real_distribution<double> dis_float(0.0, 1.0);
    uniform_int_distribution<int> dis_int(1, 1000000);

    // Έναρξη χρονομέτρησης
    auto start = chrono::high_resolution_clock::now();

    ofstream file(filename);
    
    // ΣΗΜΑΝΤΙΚΟ: Μεγάλος buffer (1MB) για να μειωθούν τα system calls στον δίσκο
    vector<char> buffer(1024 * 1024);
    file.rdbuf()->pubsetbuf(buffer.data(), buffer.size());

    file << "ID,Value1,Value2,Category\n";

    for (int i = 1; i <= rows; ++i) {
        // Χρήση '\n' αντί για endl για ταχύτητα
        file << i << "," 
             << dis_float(gen) << "," 
             << dis_int(gen) << "," 
             << "ABCDE\n";

        if (i % 5000000 == 0) {
            cout << "Πρόοδος: " << i << " γραμμές..." << endl;
        }
    }

    file.close();

    // Λήξη χρονομέτρησης
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;

    cout << "------------------------------------" << endl;
    cout << "Συνολικός χρόνος: " << fixed << setprecision(2) << diff.count() << " δευτερόλεπτα" << endl;
    cout << "Ταχύτητα: " << (rows / diff.count()) / 1e6 << " εκατ. γραμμές/δευτ." << endl;

    return 0;
}


