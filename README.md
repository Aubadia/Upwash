# Upwash

The [lab specification](docs/Spec.md) records the training target, experiment
constraints, evaluation, logging, forecasting, and submission requirements.
See the [roadmap](roadmap/README.md) for implementation progress and test evidence.

## Running tests

Prepare `.venv` with the project dependencies, including pytest, then run:

```bash
# Run all tests.
./test.sh

# Run one test file.
./test.sh tests/unit/core/test_trainer_loss.py

# Select tests by name and show detailed output.
./test.sh -k compute_loss -v
```

The script uses `.venv/bin/python`, makes `src/` importable, and forwards
arguments to pytest. It does not install dependencies. Virtual environment
activation is not required. Test paths are relative to the project root, even
when invoking the script from another directory. The script returns pytest's
exit status.
