# NeuroMotion-HAR — Complete Project Structure

```
NeuroMotion-HAR/                          # 128+ files, 98 directories
│
├── 📄 README.md                          # 600+ line exceptional README
├── 📄 LICENSE                            # MIT License
├── 📄 CONTRIBUTING.md                    # Contribution guide
├── 📄 CODE_OF_CONDUCT.md                 # Community standards
├── 📄 SECURITY.md                        # Vulnerability reporting
├── 📄 CHANGELOG.md                       # Version history
├── 📄 CITATION.cff                       # Academic citation
├── 📄 .gitignore                         # Git exclusions
├── 📄 .pre-commit-config.yaml            # Black + Ruff + mypy hooks
├── 📄 pyproject.toml                     # Python package config
├── 📄 requirements.txt                   # Production dependencies
├── 📄 requirements-dev.txt               # Dev + testing deps
├── 📄 .env.example                       # Environment template
├── 📄 docker-compose.yml                 # Full stack orchestration
├── 📄 PROJECT_STRUCTURE.md               # This file
│
├── 📦 backend/                           # FastAPI application
│   ├── alembic.ini                       # DB migrations config
│   └── app/
│       ├── main.py                       # FastAPI entry point
│       ├── api/v1/endpoints/
│       │   ├── auth.py                   # JWT auth (register/login)
│       │   ├── predict.py                # REST + WebSocket inference
│       │   ├── health.py                 # Health intelligence API
│       │   ├── activities.py             # Activity history
│       │   ├── twin.py                   # Digital twin API
│       │   ├── sensors.py                # Sensor management
│       │   └── anomaly.py                # Anomaly detection API
│       ├── core/
│       │   ├── config.py                 # Pydantic settings
│       │   ├── security.py               # JWT + password hashing
│       │   └── logging.py                # Loguru configuration
│       ├── db/
│       │   ├── session.py                # AsyncSQLAlchemy engine
│       │   ├── models.py                 # User, Activity, Health, Anomaly
│       │   └── migrations/               # Alembic migration scripts
│       ├── ml/
│       │   ├── model_loader.py           # Model registry (startup)
│       │   ├── inference.py              # SHAP + prediction engine
│       │   └── feature_extractor.py      # 100+ hand-crafted features
│       ├── schemas/
│       │   ├── predict.py                # SensorWindow, PredictionResult
│       │   ├── auth.py                   # User schemas
│       │   └── health.py                 # Health score schemas
│       └── services/
│           ├── digital_twin.py           # Behavioral modeling engine
│           └── health_intelligence.py    # AI insight generation
│
├── 🌐 frontend/                          # Next.js 14 dashboard
│   ├── package.json                      # Node dependencies
│   ├── tsconfig.json                     # TypeScript config
│   ├── tailwind.config.ts                # Dark theme + custom tokens
│   ├── next.config.ts                    # Next.js configuration
│   └── src/
│       ├── app/
│       │   ├── layout.tsx                # Root layout + metadata
│       │   ├── globals.css               # Tailwind + custom utilities
│       │   ├── page.tsx                  # Root → redirect to dashboard
│       │   ├── dashboard/page.tsx        # Main dashboard (activity + health)
│       │   ├── analytics/page.tsx        # Charts + 4-week trends
│       │   ├── insights/page.tsx         # AI insights with evidence
│       │   └── profile/page.tsx          # Digital twin profile
│       ├── components/
│       │   ├── layout/
│       │   │   ├── Providers.tsx         # React Query + Toast providers
│       │   │   └── Sidebar.tsx           # Navigation with live indicator
│       │   ├── charts/
│       │   │   ├── ActivityCard.tsx      # Live activity + SHAP bars
│       │   │   ├── HealthScoreRing.tsx   # Animated SVG score ring
│       │   │   ├── ActivityTimeline.tsx  # Daily intensity area chart
│       │   │   ├── InsightCard.tsx       # AI insight cards
│       │   │   └── StatsGrid.tsx         # Steps/calories/HR/time
│       │   └── sensors/
│       │       └── SensorStream.tsx      # Real-time accelerometer + HR chart
│       ├── hooks/
│       │   ├── useRealtimeActivity.ts    # WebSocket prediction hook
│       │   └── useHealthScore.ts         # React Query health data
│       └── lib/
│           └── api.ts                    # Type-safe API client
│
├── 🧠 models/                            # All ML architectures
│   ├── architectures/
│   │   ├── base_model.py                 # Abstract BaseHARModel
│   │   ├── cnn_lstm.py                   # CNN-LSTM Hybrid (96.3%)
│   │   ├── patchtst.py                   # PatchTST + TCN (97.8%, 96.8%)
│   │   ├── lstm_variants.py              # LSTM, BiLSTM, GRU, AttLSTM, CNN1D, Ensemble
│   │   └── anomaly_detector.py           # LSTM Autoencoder + Isolation Forest
│   ├── configs/
│   │   ├── model_registry.yaml           # All models with benchmark results
│   │   └── training_defaults.yaml        # Default hyperparameters
│   ├── checkpoints/                      # Trained .pt files (DVC tracked)
│   └── exports/
│       ├── tflite/                       # INT8 TFLite models for RPi/Android
│       └── onnx/                         # ONNX models for cross-platform
│
├── 📊 data/                              # Data management (DVC tracked)
│   ├── README.md                         # Dataset documentation + citations
│   ├── raw/                              # Original downloaded datasets
│   ├── processed/                        # Numpy sliding windows
│   ├── external/                         # Third-party processed data
│   └── augmented/                        # Augmented training data
│
├── 📓 notebooks/                         # Jupyter research notebooks
│   ├── 01_eda_uci_har.ipynb              # EDA: signals, FFT, class balance
│   ├── 02_model_training_comparison.ipynb # All models benchmarked
│   └── 03_explainability_shap.ipynb      # SHAP beeswarm + attention maps
│
├── 🔬 training/                          # Training infrastructure
│   ├── pipelines/
│   │   ├── train.py                      # Main training CLI (all models)
│   │   ├── benchmark.py                  # Cross-model/dataset benchmark suite
│   │   └── dataset.py                    # HARDataset + augmentation
│   ├── callbacks/
│   │   ├── early_stopping.py             # Patience-based stopping
│   │   ├── model_checkpoint.py           # Best model saving
│   │   └── lr_scheduler.py               # Warmup cosine annealing
│   ├── configs/
│   │   ├── patchtst.yaml                 # PatchTST hyperparams
│   │   ├── cnn_lstm.yaml                 # CNN-LSTM hyperparams
│   │   ├── preprocess.yaml               # Window / normalization params
│   │   └── training_defaults.yaml        # Shared training config
│   └── experiments/                      # Saved experiment JSON results
│
├── 🚀 deployment/                        # All deployment configs
│   ├── docker/
│   │   ├── Dockerfile.backend            # Python FastAPI image
│   │   └── Dockerfile.frontend           # Next.js standalone image
│   ├── kubernetes/
│   │   └── deployment.yaml               # Backend, Frontend, Ingress, HPA
│   ├── cloud/
│   │   ├── aws/
│   │   │   ├── main.tf                   # ECS + RDS + ElastiCache + CloudFront
│   │   │   └── deploy.sh                 # One-command AWS deployment
│   │   ├── gcp/                          # Cloud Run deployment
│   │   └── azure/                        # ACI deployment
│   └── edge/
│       ├── raspberry_pi/
│       │   ├── inference.py              # MPU-6050 + TFLite real-time loop
│       │   └── README.md                 # Hardware wiring + setup guide
│       ├── android/                      # Android Studio project
│       └── wear_os/                      # Wear OS app
│
├── 📈 mlops/                             # MLOps tooling
│   ├── dvc/dvc.yaml                      # DVC pipeline stages
│   ├── mlflow/                           # MLflow tracking config
│   ├── wandb/sweep_config.yaml           # Bayesian hyperparameter search
│   └── airflow/
│       ├── dags/har_training_dag.py      # Weekly retraining DAG
│       └── docker-compose.airflow.yml    # Airflow deployment
│
├── 📉 monitoring/                        # Observability stack
│   ├── prometheus/prometheus.yml         # Scrape config
│   └── grafana/
│       └── dashboards/dashboard.json     # Pre-built dashboards
│
├── 🔭 research/                          # Research documentation
│   ├── literature/
│   │   └── literature_review.md          # 120+ paper review
│   ├── benchmarks/
│   │   ├── benchmark_analysis.md         # Full benchmark with ablations
│   │   └── sota_comparison.md            # SOTA comparison table
│   ├── experiments/
│   │   └── ablation_study.md             # 47 design choice ablations
│   └── papers/
│       └── future_work.md                # 10 research directions
│
├── 🎨 assets/                            # Visual assets
│   ├── README.md                         # Asset guide + royalty-free sources
│   ├── logos/
│   │   └── neuro_motion_banner.svg       # 1200×280 README banner
│   ├── diagrams/
│   │   └── system_architecture.svg       # Full system architecture
│   ├── images/                           # Screenshots (add after setup)
│   └── demo/                             # GIFs (create with LICEcap)
│
├── 🧪 tests/                             # Test suite
│   ├── ml/test_models.py                 # Model shape/gradient/speed tests
│   ├── unit/test_inference.py            # Inference engine unit tests
│   ├── integration/                      # API integration tests
│   └── e2e/locustfile.py                 # Load testing (Locust)
│
├── 📜 scripts/                           # Utility scripts
│   ├── download/download_all.py          # Multi-dataset downloader
│   ├── preprocess/preprocess_all.py      # Sliding window preprocessing
│   ├── deploy/export_tflite.py           # TFLite/ONNX/TorchScript export
│   └── quickstart.sh                     # One-command local setup
│
└── 🔄 .github/                           # GitHub configuration
    └── workflows/
        ├── ci.yml                        # Full CI: lint + test + build + release
        └── train.yml                     # Manual/scheduled model training
```

## Quick Stats
- **128 files** across 98 directories
- **10+ model architectures** benchmarked
- **5 datasets** with download + preprocessing scripts
- **5 deployment targets** (Docker, K8s, AWS, RPi, Android)
- **3 Jupyter notebooks** with publication-quality visualizations
- **Full MLOps stack**: DVC + MLflow + W&B + Airflow + Prometheus + Grafana
- **CI/CD**: GitHub Actions with lint, test, build, deploy, and scheduled training
