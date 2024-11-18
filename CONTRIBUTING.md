# Contributing to MatterTune

First off, thank you for considering contributing to MatterTune! We want to make contributing to MatterTune as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

1. Clone your fork of the repo:
   ```bash
   git clone https://github.com/Fung-Lab/mattertune.git
   ```

2. Install development dependencies:
   ```bash
   cd mattertune
   pip install -e
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

- We use [ruff](https://github.com/charliermarsh/ruff) for formatting, linting, and import sorting
- We use [pyright](https://github.com/microsoft/pyright) for type checking

Our pre-commit hooks will automatically format your code when you commit. To run formatting manually:

```bash
# Format code + imports
ruff check --select I --fix && ruff check --fix  && ruff format

# Run linting
ruff check .

# Run type checking
pyright
```

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable
2. Update the documentation with details of any new functionality
3. Add or update tests as appropriate
4. Use clear, descriptive commit messages
5. The PR should be reviewed by at least one maintainer
6. Update the CHANGELOG.md with a note describing your changes

## Testing

We use pytest for testing. To run tests:

```bash
pytest
```

For coverage report:

```bash
pytest --cov=mattertune
```

## Reporting Bugs

We use GitHub issues to track public bugs. Report a bug by [opening a new issue]().

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

We love feature requests! To submit a feature request:

1. Check if the feature has already been requested in the issues
2. If not, create a new issue with the label "enhancement"
3. Include:
   - Clear description of the feature
   - Rationale for the feature
   - Example use cases
   - Potential implementation approach (optional)

## Documentation

Documentation improvements are always welcome. Our docs are in the `docs/` folder and use Markdown format.

## License

By contributing, you agree that your contributions will be licensed under the MIT License that covers the project. Feel free to contact the maintainers if that's a concern.

## Working with Model Backbones

When working with model backbones, please note:

- JMP (CC BY-NC 4.0 License): Non-commercial use only
- EquiformerV2 (Meta Research License): Follow Meta's Acceptable Use Policy
- M3GNet (BSD 3-Clause): Include required notices
- ORB (Apache 2.0): Include required notices and attribution

## Questions?

Don't hesitate to ask questions about how to contribute. You can:

1. Open an issue with your question
2. Tag your issue with "question"
3. We'll get back to you as soon as we can

## Attribution and References

When adding new features or modifying existing ones, please add appropriate references to papers, repositories, or other sources that informed your implementation.

Thank you for contributing to MatterTune! 🎉
