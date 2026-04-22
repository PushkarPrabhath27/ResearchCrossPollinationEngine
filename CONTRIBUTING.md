# Contributing to Research Cross-Pollination Engine (RCPE)

First off, thank you for considering contributing to RCPE! It's people like you that make RCPE such a great tool for the scientific community.

RCPE is a high-performance research platform designed to break down disciplinary silos. We maintain a high bar for code quality, scientific rigor, and documentation.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Environment Setup](#development-environment-setup)
4. [Styleguides](#styleguides)
5. [Testing](#testing)
6. [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by the [RCPE Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs
* **Check the Issue Tracker**: See if the bug has already been reported.
* **Use the Template**: When opening a new issue, use the Bug Report template.
* **Be Specific**: Include logs, environment details, and a minimal reproducible example.

### Suggesting Enhancements
* **Research Justification**: For new agentic patterns or data sources, provide a brief scientific justification or reference paper.
* **Use the Template**: Use the Feature Request or Research Suggestion templates.

## Development Environment Setup

### Prerequisites
* Python 3.10+
* Node.js 18+ (for frontend)
* Docker (optional, for infrastructure testing)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PushkarPrabhath27/ResearchCrossPollinationEngine.git
   cd ResearchCrossPollinationEngine
   ```
2. Set up the Python environment:
   ```bash
   make setup
   ```
3. Configure your environment:
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

## Styleguides

### Git Commit Messages
* Use the imperative mood ("Add feature" not "Added feature")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally

### Python Styleguide
* We strictly follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
* Use type hints for all public functions and classes.
* Docstrings should follow the [Google Style Guide](https://google.github.io/styleguide/pyguide.html).
* Run `make lint` before committing.

### Documentation Styleguide
* Use Markdown for all documentation.
* Keep architecture diagrams updated using Mermaid.js syntax in `README.md`.

## Testing

We aim for high test coverage (80%+) for core RAG and agentic logic.
* Run all tests: `make test`
* Run specific tests: `pytest tests/test_agents.py`

## Pull Request Process

1. Create a new branch: `git checkout -b feature/your-feature-name`.
2. Ensure all tests pass and linting is clean.
3. Update the `README.md` or relevant documentation if you change public APIs.
4. The PR will be reviewed by at least one maintainer.
5. Once approved, it will be merged into `main`.

---

**RCPE Maintainers**
Pushkar Prabhath
