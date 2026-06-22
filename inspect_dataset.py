import csv

rows = []
with open('battery_cycle_level_dataset_CLEAN_FINAL.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f'Totale righe: {len(rows)}')
batteries = set(r['battery_id'] for r in rows)
print(f'Batterie: {sorted(batteries)}')
print(f'Colonne: {list(rows[0].keys())}')

cycles = [int(r['cycle']) for r in rows]
capacities = [float(r['capacity']) for r in rows]
sohs = [float(r['soh']) for r in rows]
ruls = [int(r['rul']) for r in rows]

print(f'Cicli: min={min(cycles)}, max={max(cycles)}')
print(f'Capacita: min={min(capacities):.4f}, max={max(capacities):.4f}, mean={sum(capacities)/len(capacities):.4f}')
print(f'SOH: min={min(sohs):.4f}, max={max(sohs):.4f}')
print(f'RUL: min={min(ruls)}, max={max(ruls)}')

# Per batteria
for b in sorted(batteries):
    b_rows = [r for r in rows if r['battery_id'] == b]
    b_cycles = [int(r['cycle']) for r in b_rows]
    b_cap = [float(r['capacity']) for r in b_rows]
    print(f'  {b}: {len(b_rows)} cicli, capacita [{min(b_cap):.3f}, {max(b_cap):.3f}]')
