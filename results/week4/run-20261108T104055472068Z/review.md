# Week 4: generated result review

Train OOF selection results; test was not loaded.
Sample SD (ddof=1) is across sampler seeds on the same split/folds.
Cost summaries are conditional on finding a feasible solution; inspect success counts.

- random: 5/5 feasible seeds; cost mean=0.7452500000000001, SD=0.028248340659231666; mean search seconds=31.656344420000096.
- tpe: 5/5 feasible seeds; cost mean=0.69625, SD=0.01224744871391587; mean search seconds=33.053856280000126.

Paired comparisons (one difference per sampler seed):
- tpe: 4 seeds.
- random: 1 seeds.

Interpretation limits:
- No claim of statistical significance from five seeds alone.
- Runtime includes log writing and can change with system load; reused seed 42 ran earlier.
- Sensitivity reruns searches per scenario with seed 42 only; other seeds are not tested for sensitivity.
- Changing FN cost changes the objective scale. Compare methods within the same scenario.
- Fixed RF baseline OOF is rescored; its threshold is selected again within each scenario.
- No final method/configuration is selected automatically. Keep the 5:1 / 35% main protocol.
