import csv
rows = list(csv.DictReader(open('output/saturation_20260314_205919/benchmark_20260315_074617/results.csv')))
sa_outliers = [r for r in rows if r['algorithm']=='sa' and float(r['composite_score']) > 0.01]
for r in sorted(sa_outliers, key=lambda x: float(x['composite_score']), reverse=True)[:10]:
    print(f"seed={r['seed']} budget={r['budget']} composite={r['composite_score']} condition={r['condition']}")
