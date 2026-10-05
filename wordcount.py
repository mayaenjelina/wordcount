
import os, sys, re, time, csv, argparse, statistics
from collections import Counter

# Pastikan worker Spark memakai Python yang sama (penting di Windows)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(line):
    """Preprocessing: lowercase, buang tanda baca/emoji, ambil kata alfanumerik."""
    return TOKEN.findall(line.lower())


def timeit(fn, repeat=3):
    """Jalankan fn beberapa kali, kembalikan (hasil terakhir, median waktu detik)."""
    times, res = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn()
        times.append(time.perf_counter() - t0)
    return res, statistics.median(times)


def single_machine(path, scale):
    """Baseline: WordCount murni Python (1 proses, 1 core)."""
    c = Counter()
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for _ in range(scale):
        for line in lines:
            c.update(tokenize(line))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="bigdata.txt")
    ap.add_argument("--scale", type=int, default=1, help="gandakan data N kali")
    ap.add_argument("--partitions", default="1,2,4,8,16,32")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    os.makedirs("hasil", exist_ok=True)

    spark = (SparkSession.builder.appName("WordCount")
             .master("local[*]")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    cores = sc.defaultParallelism
    print(f"Spark {spark.version} | local[*] -> {cores} core | scale x{args.scale}")

    # ---- Membaca data ----
    lines = sc.textFile(args.input)
    if args.scale > 1:
        s = args.scale
        lines = lines.flatMap(lambda l: [l] * s)

    def build(rdd):
        words = rdd.flatMap(tokenize)                    # transformation (narrow)
        pairs = words.map(lambda w: (w, 1))              # transformation (narrow)
        return pairs.reduceByKey(lambda a, b: a + b)     # transformation (wide / shuffle)

    # ---- 1. WordCount + Top 10 ----
    wc = build(lines)
    top10 = wc.sortBy(lambda x: -x[1]).take(10)          # action
    total_words = lines.flatMap(tokenize).count()        # action
    uniq = wc.count()                                    # action
    print(f"\nTotal baris: {lines.count():,} | total kata: {total_words:,} | kata unik: {uniq:,}")
    print("\n10 kata terbanyak:")
    for i, (w, c) in enumerate(top10, 1):
        print(f"{i:>2}. {w}\t{c:,}")
    with open("hasil/top10.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f); w_.writerow(["peringkat", "kata", "frekuensi"])
        for i, (w, c) in enumerate(top10, 1):
            w_.writerow([i, w, c])

    # ---- Lineage / DAG (bahan diagram Job -> Stage -> Task) ----
    print("\nLineage RDD (perhatikan batas ShuffledRDD = batas Stage):")
    print(wc.toDebugString().decode())
    print(f"Jumlah partition awal: {lines.getNumPartitions()}")

    # ---- 2. Single machine vs Spark ----
    _, t_single = timeit(lambda: single_machine(args.input, args.scale), args.repeat)
    _, t_spark = timeit(lambda: build(lines).count(), args.repeat)
    print(f"\nSingle machine (Python): {t_single:.3f} detik")
    print(f"PySpark local[*]       : {t_spark:.3f} detik")
    with open("hasil/perbandingan.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f); w_.writerow(["metode", "waktu_detik"])
        w_.writerow(["single_machine_python", round(t_single, 4)])
        w_.writerow(["pyspark_local", round(t_spark, 4)])

    # ---- 3. Pengaruh jumlah partition ----
    print("\nEksperimen partition:")
    rows = []
    for n in [int(x) for x in args.partitions.split(",")]:
        rdd = lines.repartition(n)
        rdd.count()                                       # warm-up
        _, t = timeit(lambda: build(rdd).count(), args.repeat)
        rows.append((n, round(t, 4)))
        print(f"  partition = {n:>2} -> {t:.3f} detik")
    best = min(rows, key=lambda r: r[1])
    with open("hasil/partition.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f); w_.writerow(["partition", "waktu_detik"]); w_.writerows(rows)
    print(f"Partition tercepat: {best[0]} ({best[1]} detik)")

    print("\nSelesai. Hasil tersimpan di folder 'hasil/'.")
    spark.stop()


if __name__ == "__main__":
    main()