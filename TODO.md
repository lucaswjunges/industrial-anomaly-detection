# TODO - Known Issues & Future Improvements

## High Priority

- [ ] Fix autoencoder threshold calibration
  - Current: using 99th percentile of training error (too conservative)
  - Better: use separate validation set for threshold tuning
  - Or: try reconstruction error + density estimation

- [ ] Add proper ablation study for features
  - Which features actually matter?
  - Are all 27 features needed or is this overkill?
  - Test with minimal feature set vs full set

- [ ] Test with more realistic anomaly rates
  - Current test: 10% anomalies
  - Real systems: 99%+ normal data
  - Need to verify FP rate doesn't explode

## Medium Priority

- [ ] Reduce LOF memory footprint
  - Currently stores all training data (~200MB for 1M samples)
  - Consider approximate KNN for larger datasets
  - Or just drop LOF entirely (not adding much value vs IF)

- [ ] Add feature importance analysis
  - SHAP values would be nice
  - Which features trigger each model's alerts?
  - Helps debug false positives

- [ ] Implement online threshold adaptation
  - Static thresholds don't handle drift well
  - Need periodic recalibration or adaptive approach

## Low Priority

- [ ] Shrink Docker image size
  - Current: ~800MB (mostly TensorFlow)
  - Switch to tensorflow-cpu or ONNX for inference?
  - Target: <400MB

- [ ] Add bearing fault frequency features
  - Current features are generic
  - Domain-specific features (BPFO, BPFI) might help
  - Requires more signal processing knowledge

- [ ] Hyperparameter tuning
  - Most params are defaults
  - Grid search or Bayesian opt might help
  - But returns are probably marginal

## Known Bugs

- [ ] Regime-aware preprocessing fails on unknown operational states
  - Throws error if test data has new regime
  - Should fall back to global scaler gracefully

## Questions / Decisions Needed

- Is ensemble voting actually helping? Should I just use the best single model?
- Do I need all 3 models or is IF + AE enough?
- Should I support streaming predictions in the API?
