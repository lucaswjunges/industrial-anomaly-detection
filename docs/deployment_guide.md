# Deployment Guide: Edge vs Cloud Architecture

## Overview

This guide provides detailed deployment strategies for the Industrial IoT Anomaly Detection system, comparing edge, cloud, and hybrid approaches with specific implementation recommendations.

---

## Architecture Options

### Option 1: Edge Deployment

**Target Use Case:** Facilities requiring low-latency detection with minimal cloud dependency

#### Hardware Specifications

**Recommended Platforms:**

1. **NVIDIA Jetson Nano (Preferred)**
   - GPU acceleration for Autoencoder inference
   - 4GB RAM, quad-core ARM CPU
   - Power consumption: 5-10W
   - Cost: ~$99

2. **Raspberry Pi 4 (Budget Option)**
   - 8GB RAM model recommended
   - Suitable for Isolation Forest only
   - Power consumption: 3-5W
   - Cost: ~$75

3. **Industrial Edge Gateway (Production)**
   - Moxa UC-8112-LX / Advantech UNO-2483G
   - Wide temperature range (-40°C to 75°C)
   - DIN-rail mounting
   - Cost: $400-800

#### Software Stack

```
Operating System: Ubuntu 20.04 LTS (ARM64)
Runtime: Python 3.8 + TensorFlow Lite
Message Queue: Mosquitto MQTT
Time-series DB: SQLite (local buffer)
Visualization: Grafana + InfluxDB (optional)
```

#### Data Flow

```
┌─────────────┐
│  6 Sensors  │ (Modbus RTU / 4-20mA)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  PLC / Data Logger  │ (1-min sampling)
└──────┬──────────────┘
       │ MQTT
       ▼
┌──────────────────────────────────────┐
│        Edge Gateway (Jetson Nano)    │
│  ┌────────────────────────────────┐  │
│  │  Preprocessing                 │  │
│  │  - Normalization (regime-aware)│  │
│  │  - Feature engineering         │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │  Anomaly Detection             │  │
│  │  - Isolation Forest (primary)  │  │
│  │  - Autoencoder (secondary)     │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │  Alert Logic                   │  │
│  │  - Threshold filtering         │  │
│  │  - De-duplication (5-min window│  │
│  └────────────────────────────────┘  │
└──────┬───────────────────────────────┘
       │
       ├──▶ Local HMI (operator display)
       │
       └──▶ Cloud (hourly sync, optional)
```

#### Performance Characteristics

| Model | Inference Time | CPU Usage | Memory | Power |
|-------|----------------|-----------|--------|-------|
| Isolation Forest | 12 ms | 15% | 250 MB | 0.5 W |
| LOF | 45 ms | 22% | 400 MB | 0.8 W |
| Autoencoder (CPU) | 150 ms | 55% | 600 MB | 2.0 W |
| Autoencoder (GPU) | 8 ms | 10% | 800 MB | 3.5 W |

**Recommendation:** Isolation Forest for latency-critical applications, Autoencoder on GPU for best accuracy.

#### Installation Steps

```bash
# 1. Flash Ubuntu 20.04 to SD card
# 2. Initial setup
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git mosquitto -y

# 3. Clone project
git clone https://github.com/yourusername/iot-anomaly-detection.git
cd iot-anomaly-detection

# 4. Install TensorFlow Lite (ARM optimized)
pip3 install https://github.com/PINTO0309/TensorflowLite-bin/releases/download/v2.9.0/tflite_runtime-2.9.0-cp38-cp38-linux_aarch64.whl

# 5. Install dependencies
pip3 install -r requirements_edge.txt

# 6. Convert models to TFLite
python3 scripts/convert_to_tflite.py

# 7. Configure MQTT connection
nano config/edge_config.yaml

# 8. Test inference
python3 src/edge_inference.py --test

# 9. Set up systemd service (auto-start)
sudo cp deployment/edge-anomaly-detector.service /etc/systemd/system/
sudo systemctl enable edge-anomaly-detector
sudo systemctl start edge-anomaly-detector
```

#### Monitoring & Maintenance

**Local Monitoring:**
- Watchdog timer (restart on crash)
- Daily health check (sensor connectivity, model latency)
- Weekly reboot (Sunday 2 AM maintenance window)

**Remote Access:**
- VPN tunnel for secure SSH access
- TeamViewer / AnyDesk for HMI screen sharing

**Model Updates:**
- Download new models from S3/FTP during maintenance window
- A/B testing: run old + new models in parallel for 24 hours
- Automatic rollback if FP rate exceeds threshold

---

### Option 2: Cloud Deployment

**Target Use Case:** Multi-site operations, centralized analytics, rapid iteration

#### Architecture

```
┌──────────────┐
│ Facility 1-N │ (Edge gateway with basic data acquisition)
└──────┬───────┘
       │ TLS/HTTPS
       ▼
┌─────────────────────────────────────────────┐
│            Cloud Platform (AWS)             │
│  ┌─────────────────────────────────────┐   │
│  │  IoT Core (MQTT broker)             │   │
│  └──────┬──────────────────────────────┘   │
│         │                                    │
│         ▼                                    │
│  ┌─────────────────────────────────────┐   │
│  │  Kinesis Data Streams               │   │
│  │  (real-time ingestion)              │   │
│  └──────┬──────────────────────────────┘   │
│         │                                    │
│         ├─▶ Lambda (preprocessing)          │
│         │                                    │
│         ▼                                    │
│  ┌─────────────────────────────────────┐   │
│  │  SageMaker Endpoint                 │   │
│  │  - Autoencoder model                │   │
│  │  - Auto-scaling (2-10 instances)    │   │
│  └──────┬──────────────────────────────┘   │
│         │                                    │
│         ▼                                    │
│  ┌─────────────────────────────────────┐   │
│  │  DynamoDB (anomaly events)          │   │
│  │  TimeStream (time-series data)      │   │
│  └─────────────────────────────────────┘   │
│                                              │
│  ┌─────────────────────────────────────┐   │
│  │  SNS (alerts)                       │   │
│  │  - Email, SMS, PagerDuty            │   │
│  └─────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

#### Cost Estimate (1 facility, 12 pumps)

**Monthly AWS Costs:**
- IoT Core: 12 devices × $0.08/device = $0.96
- Data transfer: 60 MB/day × $0.09/GB = $0.16
- SageMaker inference: 500K requests × $0.0001 = $50
- Lambda: 1M invocations × $0.20/1M = $0.20
- DynamoDB: 1 GB storage + 100K writes = $1.50
- TimeStream: 10 GB storage × $0.50/GB = $5.00
- CloudWatch: $5.00

**Total:** ~$63/month per facility

**Scaling:**
- 10 facilities: ~$500/month
- 100 facilities: ~$3,800/month (volume discounts apply)

#### Advantages

1. **Centralized Management:** One dashboard for all facilities
2. **Rapid Updates:** Deploy new models in minutes
3. **Advanced Analytics:** Cross-facility pattern analysis
4. **Unlimited Compute:** Scale to handle load spikes
5. **Data Retention:** 7-year compliance storage

#### Disadvantages

1. **Latency:** 500-1500ms end-to-end (network dependent)
2. **Connectivity Dependency:** Requires stable internet
3. **Recurring Costs:** $500-5000/month depending on scale
4. **Data Privacy:** Sensor data leaves facility

---

### Option 3: Hybrid Deployment (Recommended)

**Best of Both Worlds:** Edge for real-time, cloud for intelligence

#### Architecture

```
┌─────────────────────────────────────────────┐
│              Edge Gateway                   │
│  ┌─────────────────────────────────────┐   │
│  │  Lightweight Model (Isolation Forest)│   │
│  │  - Inference: <20ms                 │   │
│  │  - Local alerting                   │   │
│  └──────┬──────────────────────────────┘   │
│         │                                    │
│         ├─▶ Immediate alert (< 1 sec)       │
│         │                                    │
│         ▼                                    │
│  ┌─────────────────────────────────────┐   │
│  │  Data Buffer (5-min window)         │   │
│  └──────┬──────────────────────────────┘   │
└─────────┼────────────────────────────────────┘
          │
          │ Encrypted tunnel
          ▼
┌─────────────────────────────────────────────┐
│              Cloud Platform                 │
│  ┌─────────────────────────────────────┐   │
│  │  Deep Model (Autoencoder)           │   │
│  │  - Anomaly confirmation             │   │
│  │  - Classification (failure type)    │   │
│  └─────────────────────────────────────┘   │
│                                              │
│  ┌─────────────────────────────────────┐   │
│  │  Analytics & Retraining             │   │
│  │  - Model performance monitoring     │   │
│  │  - Monthly retraining               │   │
│  │  - Push updates to edge             │   │
│  └─────────────────────────────────────┘   │
└──────────────────────────────────────────────┘
```

#### Workflow

1. **Edge Detection (Real-time)**
   - Isolation Forest runs on every sample (1-min)
   - If anomaly score > threshold → immediate local alert
   - HMI displays: "Potential anomaly detected - cloud analyzing"

2. **Cloud Confirmation (15-30 sec)**
   - Edge sends 5-minute buffer to cloud
   - Autoencoder analyzes full context
   - Classification model identifies failure mode
   - Cloud sends back: "Confirmed: Cavitation - Priority HIGH"

3. **Operator Action**
   - Operator receives enriched alert with diagnosis
   - Access cloud dashboard for historical context
   - Log action taken → feeds back to retraining

#### Failover Strategy

**If cloud unavailable:**
- Edge continues autonomous operation
- Local SQLite buffer stores data (7 days capacity)
- Operator relies on edge alerts only
- Auto-sync when connection restored

**If edge fails:**
- Cloud monitors heartbeat (5-min interval)
- Sends SMS alert to maintenance team
- Historical data used to flag high-risk periods

---

## Security Considerations

### Edge Security

1. **Physical Security**
   - Locked enclosure for edge hardware
   - Tamper detection sensors
   - Read-only filesystem (prevent malware)

2. **Network Security**
   - Firewall: allow only outbound HTTPS/MQTT
   - VPN for remote access (WireGuard/OpenVPN)
   - Certificate-based authentication

3. **Application Security**
   - Code signing for model updates
   - Encrypted storage for credentials
   - Principle of least privilege (no root access)

### Cloud Security

1. **Data in Transit**
   - TLS 1.3 for all connections
   - Mutual TLS (device certificates)

2. **Data at Rest**
   - AES-256 encryption for S3/DynamoDB
   - KMS key rotation (90 days)

3. **Access Control**
   - IAM roles with MFA
   - Audit logging (CloudTrail)
   - Network segmentation (VPC, security groups)

---

## Recommended Deployment Path

### Phase 1: Proof of Concept (1-2 months)

**Setup:**
- Deploy edge gateway on 2 pumps
- Cloud analytics in parallel (shadow mode)
- Human verification of all alerts

**Success Criteria:**
- <5% false positive rate
- Detect 2+ real anomalies
- Operator feedback positive

### Phase 2: Pilot (2-3 months)

**Setup:**
- Expand to 6 pumps (50% of facility)
- Active alerting enabled
- Daily review meetings

**Success Criteria:**
- Prevent 1+ unplanned shutdown
- <3 false alarms/day per operator
- 80%+ detection rate

### Phase 3: Production (1 month)

**Setup:**
- Full facility rollout (12 pumps)
- Integration with CMMS (work orders)
- Automated reporting

**Success Criteria:**
- 15%+ reduction in unplanned downtime
- ROI > 3x in first year
- Operator acceptance

### Phase 4: Optimization (Ongoing)

- Monthly model retraining with labeled data
- Quarterly hyperparameter tuning
- Expand to other equipment types

---

## Troubleshooting

### Common Issues

**Issue:** High false positive rate after deployment

**Solutions:**
1. Check operational state labeling (startup vs. normal)
2. Retrain with site-specific data (transfer learning)
3. Increase anomaly threshold (trade recall for precision)
4. Add time-of-day filters (expected high-load periods)

**Issue:** Missed anomalies (low recall)

**Solutions:**
1. Review missed events for common patterns
2. Add targeted features (e.g., pressure-flow ratio for cavitation)
3. Reduce threshold (accept more false positives)
4. Ensemble multiple models (OR logic for high-severity alerts)

**Issue:** Edge device crashes

**Solutions:**
1. Check memory usage (reduce batch size)
2. Implement watchdog timer (auto-restart)
3. Update to lightweight model (Isolation Forest only)
4. Monitor temperature (throttling on Jetson)

---

## Maintenance Checklist

### Daily
- [ ] Review alert log (false positive patterns)
- [ ] Verify sensor connectivity (all 6 signals)

### Weekly
- [ ] Check model inference latency
- [ ] Review cloud sync status
- [ ] Backup local data buffer

### Monthly
- [ ] Model performance review (precision/recall trends)
- [ ] Retrain models with new labeled data
- [ ] Update edge devices (security patches)

### Quarterly
- [ ] Hardware health check (SD card, power supply)
- [ ] Disaster recovery test (edge failover)
- [ ] Operator training refresher

---

## Conclusion

**For most industrial facilities, we recommend the Hybrid approach:**

- **Edge:** Isolation Forest on Jetson Nano ($99 hardware)
- **Cloud:** AWS SageMaker for Autoencoder + analytics (~$100/month)
- **Total Cost:** ~$1,500 upfront + $100/month/facility
- **Expected ROI:** 8-15x in year one

This balances low-latency real-time detection with the intelligence and flexibility of cloud analytics, while maintaining graceful degradation if either component fails.
