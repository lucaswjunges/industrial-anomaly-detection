# Technical Notes & Design Decisions

## Motivation

Started this project to learn production ML engineering practices beyond just training models. Wanted to understand:
- How to properly test ML code
- Deployment considerations (Docker, API)
- Working with real vs synthetic data
- Trade-offs between different anomaly detection approaches

## Design Decisions

### Why NASA Bearing Dataset?

**Rationale:**
- Needed real industrial data to understand actual ML challenges
- Public domain, well-documented dataset
- Run-to-failure experiments = clear anomaly labels
- Small enough to work with locally (~2GB)

**Trade-offs:**
- Only 3 test runs, limited failure modes
- Older dataset (2003), sensor tech has evolved
- Single bearing type, doesn't generalize to other equipment
- No environmental factors (temperature, humidity variations)

**What I'd do differently:**
- Combine multiple public datasets (NASA + CWRU + custom)
- Add data augmentation to handle limited samples
- Consider semi-supervised approach (most real deployments have few labeled failures)

### Model Selection: Why 3 Algorithms?

**Isolation Forest:**
- Fast inference (<20ms), works on CPU
- Good for "global" anomalies (things very different from normal)
- Struggles with "local" anomalies (subtle degradation in specific operational regimes)

**Local Outlier Factor:**
- Better at context-dependent anomalies
- But: slower inference (~100ms), needs careful neighbor selection
- In practice: would probably skip this, IF + Autoencoder covers most cases

**Autoencoder:**
- Can learn complex patterns, good recall potential
- Biggest issue: threshold calibration is a pain
- Current implementation uses 99th percentile of training error → probably too conservative
- Should implement dynamic thresholding or use validation set

**Honest assessment:**
- Having all 3 is a bit overkill for a real deployment
- In production, I'd probably go with just IF + ensemble voting if needed
- LOF was interesting to implement but adds complexity without huge gains

### Feature Engineering Approach

**What I did:**
- Rolling statistics (5-min windows)
- Temporal derivatives
- Frequency domain features (FFT energy, kurtosis)

**Why 5-minute windows?**
- Honestly, mostly trial-and-error
- Too short (1 min) = noisy, too long (15 min) = miss transients
- Should have done proper ablation study but... didn't

**What's missing:**
- No wavelet features (would probably help for vibration)
- No domain-specific features (bearing fault frequencies)
- Could calculate envelope spectrum for bearing diagnostics
- Feature selection is pretty naive (just threw everything in)

### Regime-Aware Preprocessing

**The idea:**
- Different operational states (startup, normal, high-load) have different baseline behavior
- Normalize separately per regime to avoid false alarms during transitions

**Does it work?**
- Yes for reducing FPs during state changes
- But: requires knowing operational state (not always available in real systems)
- Alternative: online adaptive normalization, but adds complexity

**Real-world concern:**
- What if you see a new operational regime at test time?
- Current code would fail or fall back to global scaler
- Should handle this more gracefully

## Known Limitations

### Data Issues

1. **NASA dataset limitations:**
   - Only outer race and inner race failures
   - No contamination failures (dirt, moisture)
   - Accelerated life testing ≠ real operating conditions
   - Single speed, constant load (real motors vary)

2. **Class imbalance:**
   - 90% normal, 10% anomaly
   - Real systems are way more imbalanced (99.9%+)
   - Should test with more realistic ratios

### Model Limitations

1. **Autoencoder threshold:**
   - Current approach (99th percentile) is too simplistic
   - Needs proper validation set to tune
   - Or better: use reconstruction error + density estimation

2. **No online learning:**
   - Models are static after training
   - Real degradation patterns might drift
   - Should add periodic retraining or online updates

3. **Feature drift:**
   - No monitoring for distribution shift
   - In production, would need to track feature distributions
   - Alert if test data looks very different from training

### Deployment Concerns

1. **Inference latency:**
   - IF: <20ms ✓
   - LOF: ~100ms (acceptable for 1-min sampling)
   - Autoencoder: ~50ms on CPU (would be faster on GPU)
   - But: feature engineering adds ~30ms
   - Total: ~150ms for ensemble, probably fine for non-real-time

2. **Memory footprint:**
   - LOF stores all training data (not scalable)
   - For 1M samples × 27 features × 8 bytes ≈ 200MB just for LOF
   - Would need to switch to approximate KNN for larger datasets

3. **Docker image size:**
   - Current: ~800MB (could be smaller)
   - Mostly TensorFlow (500MB)
   - Could use tensorflow-cpu for smaller image
   - Or switch to ONNX for inference only

## What I'd Improve Given More Time

### High Priority

1. **Better threshold calibration:**
   - Use proper validation set
   - Try different percentiles
   - Compare against supervised alternatives (one-class SVM)

2. **Ablation studies:**
   - Which features actually matter?
   - What's the minimum viable feature set?
   - Are all 3 models needed?

3. **More realistic evaluation:**
   - Test with different anomaly rates (99%+ normal)
   - Add concept drift simulation
   - Measure degradation detection lag (not just binary classification)

### Medium Priority

4. **Better feature engineering:**
   - Domain-specific bearing fault frequencies
   - Wavelet decomposition
   - Proper feature selection (not just throw everything in)

5. **Model interpretability:**
   - SHAP values for feature importance
   - Which features trigger each alert?
   - Helps with debugging false positives

6. **Production monitoring:**
   - Log feature distributions
   - Track model drift
   - A/B testing framework

### Low Priority (Nice to Have)

7. **Hyperparameter tuning:**
   - Current params are mostly defaults
   - Should do proper grid search or Bayesian optimization
   - But: returns might be marginal

8. **More models:**
   - Try LSTM for sequence modeling
   - One-class SVM
   - Gaussian Mixture Models
   - But: diminishing returns, adds complexity

## Questions I Still Have

1. **Does ensemble voting actually help?**
   - Currently just using majority vote (2/3)
   - Should I weight by model confidence?
   - Or just pick the single best model?

2. **Is 80% test coverage enough?**
   - Current tests cover happy path well
   - Missing: adversarial inputs, edge cases
   - Should add property-based testing?

3. **Docker vs direct deployment?**
   - Docker adds overhead (~100MB image size)
   - For edge deployment, might be too heavy
   - Alternative: compiled binary (PyInstaller)?

4. **API design:**
   - Should I support streaming predictions?
   - Batch API is more efficient, but less flexible
   - Trade-off: latency vs throughput

## Lessons Learned

1. **Real data is messy:**
   - NASA dataset has outliers, sensor glitches
   - Had to add robust preprocessing
   - Synthetic data didn't prepare me for this

2. **Thresholds are hard:**
   - Spent way too much time tuning autoencoder threshold
   - Still not happy with it
   - Probably should have used a simpler model

3. **Testing ML code is different:**
   - Can't just check exact outputs (randomness)
   - Need statistical tests, property-based tests
   - Still learning best practices here

4. **Documentation takes time:**
   - Spent ~30% of time on docs/setup
   - Worth it for portfolio, but in prod I'd focus more on code

## References & Learning Resources

Books/papers that helped:
- Hands-On Machine Learning (Géron) - general ML
- Streaming Systems (Akidau) - for thinking about online deployment
- sklearn docs - API design patterns
- Various Stack Overflow threads on bearing fault detection

Still need to read:
- ML Engineering book by Andriy Burkov
- Building Machine Learning Powered Applications (Ameisen)
- Designing Data-Intensive Applications (Kleppmann)
