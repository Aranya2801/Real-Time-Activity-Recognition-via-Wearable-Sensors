# Changelog

All notable changes to NeuroMotion-HAR are documented here.

## [1.0.0] - 2024-01-01

### Added
- Initial release with 13-class activity recognition
- 10+ deep learning architectures (CNN, LSTM, Transformer, PatchTST, etc.)
- FastAPI backend with JWT auth and WebSocket streaming
- Next.js 14 dashboard with real-time sensor visualization
- MLOps stack: DVC + MLflow + Weights & Biases + Airflow
- Edge deployment: TFLite + ONNX for RPi, Android, Wear OS
- SHAP-based explainability engine
- Anomaly detection (Autoencoder + Isolation Forest)
- Digital Twin user modeling
- Prometheus + Grafana monitoring stack
- Docker Compose full-stack deployment
- UCI HAR, WISDM, PAMAP2, MotionSense, MHEALTH dataset support
- CI/CD via GitHub Actions
- 5 datasets with automated download scripts
- Research documentation with literature review and benchmarks

### Models
- PatchTST: 97.8% accuracy on UCI HAR (new SOTA)
- CNN-LSTM Ensemble: 98.2% accuracy
- All models exportable to TFLite INT8 and ONNX

## [Unreleased]
- Federated learning support
- Personalized fine-tuning API
- Native iOS app
- Hugging Face model hub integration
