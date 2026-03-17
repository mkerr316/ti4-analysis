import csv
rows = list(csv.DictReader(open('output/saturation_20260314_205919/benchmark_20260315_074617/results.csv')))
seed37 = [r for r in rows if r['seed']=='37']
for r in sorted(seed37, key=lambda x: (x['algorithm'], int(x['budget']))):
    print(f"algo={r['algorithm']:6s} budget={r['budget']:6s} composite={r['composite_score']:8s} morans_i={r['morans_i']}")
