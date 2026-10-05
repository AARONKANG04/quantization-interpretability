"""Print the layer ranking from results/phase1/ranked_order.json."""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/phase1/ranked_order.json"))
base = d["kl_all_quantized"]
print("KL all-quantized:", round(base, 4))
for r in d["ranking"][:12]:
    print(f"  layer {r['layer']:2d}  kl_restored {r['kl_restored']:.4f}  recovery {r['recovery']:+.4f}  ({100 * r['recovery'] / base:.1f}% of KL)")
tot = sum(r["recovery"] for r in d["ranking"])
print("sum of single-layer recoveries / total KL:", round(tot / base, 3))
print("bottom 3:", [(r["layer"], round(100 * r["recovery"] / base, 1)) for r in d["ranking"][-3:]])
