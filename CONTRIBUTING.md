# Contributing to Vireo

Thank you for your interest in contributing to Vireo! This document provides guidelines and information for contributors.

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Development Setup

### Prerequisites
- Rust 1.70+ and Cargo
- GPU with WebGPU support (Vulkan, Metal, or DirectX 12)
- Git

### Building
```bash
git clone <repository-url>
cd vireo
cargo build --release
```

### Running
```bash
cargo run --release
```

## Contribution Guidelines

### Code Style
- Follow Rust formatting standards (run `cargo fmt`)
- Use meaningful variable and function names
- Add comments for complex logic
- Keep functions focused and reasonably sized

### Testing
- Add tests for new functionality
- Ensure existing tests pass (`cargo test`)
- Test on multiple platforms when possible

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit with clear, descriptive messages
5. Push to your fork
6. Open a Pull Request with a clear description

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests

## Areas for Contribution

### High Priority
- Performance optimizations
- Bug fixes and stability improvements
- Documentation improvements
- Cross-platform compatibility

### Medium Priority
- New simulation features
- UI/UX improvements
- Additional shader effects
- Configuration options

### Low Priority
- Experimental features
- Alternative rendering backends
- Platform-specific optimizations

## Questions or Issues?

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Join our community channels (if available)

Thank you for contributing to Vireo!
