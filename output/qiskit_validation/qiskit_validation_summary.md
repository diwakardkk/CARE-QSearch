# Qiskit Small-N Resource Validation

Purpose: sanity-check heuristic oracle-resource estimates against Qiskit-transpiled small-N compiled phase oracles.
Scope: N in {8, 16, 32}; methods: Standard, CARE-adaptive, CARE-Fuse-95.

Validated cases: 96
Failed/skipped cases: 0

Median relative errors by method:
status                method  validated_cases  heuristic_depth_mean  qiskit_depth_mean  depth_relative_error_median  heuristic_CX_mean  qiskit_CX_mean  CX_relative_error_median  transpilation_time_mean
  pass          CARE-Fuse-95               32                  72.0          210.59375                    -0.447853             45.375        120.8125                 -0.353325                 0.001926
  pass CARE-QSearch-adaptive               32                  72.0          210.59375                    -0.447853             45.375        120.8125                 -0.353325                 0.002075
  pass       standard_grover               32                  85.0          250.09375                    -0.450777             53.375        143.1250                 -0.358730                 0.002811
