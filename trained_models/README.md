# Trained models for the KSOS Go1 box push

- `flow_lse01/`: the learned sampler, a conditional RealNVP over the 8 knots (forward, turn) of a plan,
  trained by `legged_mppi/scripts/ksos_train_flow.py` (`config.json` has the settings; its paths point to
  the original training logs).
- `sdp_fixed1/`: the learned KSOS SDP solver (GNN warm start + 3 unrolled Newton layers) for the
  fixed-length-scale SDP with lambda 0.1, trained by `legged_mppi/scripts/ksos_train_sdp.py`.
  `report.json` is its test-set comparison with Newton's method, `train.txt` the training log.

Both need `ksos_l2o` from Global-MPPI next to the installed `global_mppi` package.

Setup and the closed-loop test: see "KSOS Global Planner for Box Pushing" in the main
[README](../README.md#ksos-global-planner-for-box-pushing).
