# julia 1.7 secs
# java 2.9 secs
# C++ 1.65 secs

using CSV, Tables, Random, BenchmarkTools

function fast_generate_csv(filename, rows)
    # Ορίζουμε τους τύπους δεδομένων για να βοηθήσουμε τον compiler
    ids = 1:rows
    val1 = rand(Float64, rows)
    val2 = rand(Int32, rows)
    # Δημιουργούμε ένα string buffer για σταθερή κατηγορία (γλιτώνει allocation)
    cat = fill("ABCDE", rows)

    # Χρήση Tables.table για ελάχιστο overhead μνήμης
    tbl = (ID=ids, Value1=val1, Value2=val2, Category=cat)

    println("Έναρξη εγγραφής...")
    # Το CSV.write χρησιμοποιεί αυτόματα threads αν είναι διαθέσιμα
    @time CSV.write(filename, tbl; threaded=true)
end

# Εκτέλεση για 25 εκατομμύρια γραμμές
fast_generate_csv("large_data_julia.csv", 25_000_000)

