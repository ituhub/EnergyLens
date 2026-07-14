import sqlite3
import csv

conn = sqlite3.connect('data/energylens.db')
cursor = conn.execute('SELECT * FROM spot_prices ORDER BY valid_time')
rows = cursor.fetchall()
cols = [d[0] for d in cursor.description]

with open('training_data_fresh.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(cols)
    w.writerows(rows)

print(f'Exported {len(rows)} rows to training_data_fresh.csv')
conn.close()
