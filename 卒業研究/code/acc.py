# acc.py
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
p = df['result'].mean()
print(f"正解率: {p:.4f} ({p*100:.2f}%)")
