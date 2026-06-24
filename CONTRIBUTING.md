# Contributing to NeuroMotion-HAR

Thank you for your interest in contributing! This project welcomes contributions of all kinds.

## Ways to Contribute
- 🐛 Bug reports and fixes
- 🧠 New model architectures
- 📊 New dataset integrations
- 📖 Documentation improvements
- 🧪 Additional tests
- 🌐 Translations

## Development Setup

```bash
git clone https://github.com/yourusername/NeuroMotion-HAR.git
cd NeuroMotion-HAR
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Write tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Run pre-commit: `pre-commit run --all-files`
6. Commit with conventional commits: `git commit -m "feat: add amazing feature"`
7. Push and open a PR

## Code Style
- Python: Black + Ruff (enforced via pre-commit)
- TypeScript: ESLint + Prettier
- Docstrings: Google style
- Type hints: Required for all public functions

## Adding a New Model Architecture

1. Create `models/architectures/your_model.py`
2. Inherit from `BaseHARModel`
3. Register in `models/configs/model_registry.yaml`
4. Add benchmark entry in `training/pipelines/benchmark.py`
5. Write model tests in `tests/ml/`
6. Update `README.md` benchmark table

## Commit Convention
```
feat:     New feature
fix:      Bug fix
docs:     Documentation
test:     Tests
refactor: Code refactoring
perf:     Performance improvement
ci:       CI/CD changes
```

## Code of Conduct
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
