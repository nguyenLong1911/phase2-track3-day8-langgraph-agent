import json
d = json.load(open("outputs/hidden_metrics.json"))
print("SUMMARY:")
for k, v in d.items():
    if k != "scenario_metrics":
        print(f"  {k}: {v}")
print()
print("PER-SCENARIO:")
hdr = f'{"id":<14}{"exp":<14}{"act":<14}{"ok":<6}{"nodes":<7}{"retry":<7}{"intr":<6}{"appr_obs":<10}'
print(hdr)
print("-" * len(hdr))
for m in d["scenario_metrics"]:
    print(
        f'{m["scenario_id"]:<14}{m["expected_route"]:<14}{m["actual_route"]:<14}'
        f'{str(m["success"]):<6}{m["nodes_visited"]:<7}{m["retry_count"]:<7}'
        f'{m["interrupt_count"]:<6}{str(m["approval_observed"]):<10}'
    )
