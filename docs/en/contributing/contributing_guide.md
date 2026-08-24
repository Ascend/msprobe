# Contributing to MindStudio Probe

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-11T02:41:31.264Z pushedAt=2026-08-11T02:51:07.147Z -->

Thank you for considering contributing to MindStudio Probe (msProbe)! We welcome all forms of contributions, including bug fixes, feature enhancements, documentation improvements, and even just feedback. Whether you are an experienced developer or participating in an open-source project for the first time, your help is invaluable.

You can support this project in several ways:

- Report [issues](https://gitcode.com/Ascend/msprobe/issues).
- Suggest or implement new features.
- Improve or expand documentation.
- Review PRs and assist other contributors.
- Promote the project: share msProbe in blog posts or on social media, or star the repository.

## Finding Issues to Contribute To

Want to start contributing? Check out the following types of issues:

- [Good first issues](https://gitcode.com/Ascend/msprobe/issues?categorysearch=%255B%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%9C%80%E8%BF%91%E5%88%9B%E5%BB%BA%22%257D,%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:22797,%22name%22:%22good-first-issue%22%257D%255D,%22label%22:%22good-first-issue%22%257D%255D&state=opened&order_by=created_at&sort=desc&scope=all&page=1)

Additionally, you can explore the project's development plans and roadmap by browsing the [Issues List](https://gitcode.com/Ascend/msprobe/issues).

## Contribution Process

### Environment Requirements

- For hardware environment, see *[Ascend Product Form Description](https://www.hiascend.com/document/detail/en/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)*
- The open-source version of CANN must be installed in advance.
- Python 3.7.5 or later

### Development and Testing

1. Fork the source code to your personal repository

2. Clone the forked code from your personal repository to your local environment for code development.

   ```text
   git clone https://gitcode.com/<your-username>/msprobe.git
   cd msprobe
   ```

   Follow the [Coding Standards](#coding-standards) during code development.

3. Conduct code testing.

   See [Code Testing](#code-testing).

4. Conduct build testing.

   Compile the developed code into a whl package for testing. For detailed steps, see *[msProbe Installation Guide](../install_guide/msprobe_install_guide.md)*.

5. Develop documentation.

   If the change involves adding, modifying, or removing features, please provide relevant documentation. For detailed documentation writing requirements, see [Documentation Development](#documentation-development).

6. Submit a PR.

   See [PR Submission Process](#pr-submission-process).

### Coding Standards

#### Python Coding Standards

- Follow the PEP 8 coding style guide.
- Use 4 spaces for indentation.
- Use CamelCase for class names (e.g., `DataManager`).
- Use lowercase with underscores for function and variable names (e.g., `parse_data`).
- Add necessary type annotations and doc strings.

#### C++ Coding Standards

- Follow the existing coding style of the project
- Use 4 spaces for indentation
- Use PascalCase for class names
- Use camelCase for function names
- Add necessary comments to explain complex logic.

### Code Testing

#### Running Tests

Before submitting code, ensure all tests pass:

```bash
# Python unit tests
cd test/msprobe_test
bash run_test.sh
```

#### Adding Tests

- Add corresponding unit tests for new features
- Ensure tests cover the main logical branches
- Test cases should be readable and maintainable
- Test data should be placed in the appropriate location under the `test/` directory

#### Code Coverage

After running the tests, the code coverage report is generated in the `./report` directory.

### Documentation Development

#### Documentation Paths

If your changes affect how users interact with the product, please update the relevant documentation:

- User Guide: `docs/zh/`
- API Documentation: docstrings in code comments
- Sample Code: `samples/`

#### Documentation Standards

- Use concise and clear language
- Provide complete sample code
- Include necessary screenshots or diagram descriptions
- Ensure link validity

### PR Submission Process

#### Pre-Submission Checklist

Before submitting a PR, please ensure that:

- [ ] The code adheres to the project's Coding Standards
- [ ] Necessary test cases have been added
- [ ] All tests pass
- [ ] Updated relevant documentation
- [ ] Commit message is clear and explicit
- [ ] Code has been self-reviewed

#### Submission Process

1. **Create a Branch**

   ```bash
   git checkout -b feature/<your-feature-name>
   ```

2. **Commit Changes**

   ```bash
   git add .
   git commit -m "feat: <your feature description>"
   ```

3. **Push to Remote Repository**

   ```bash
   git push origin feature/<your-feature-name>
   ```

4. **Create a PR**

   Create a PR on GitCode and fill in:

   1. A clear title

      Follow the [Commit Message Specification](#commit-message-specification) specification.

   2. A detailed description

      Include the changes made, reasons, test results, etc.

   3. Associate the related Issue.

5. **Code Review**

   1. After submitting the PR, you need to notify the relevant Assignees (Reviewers and Committers) to review the content.

   2. You need to modify the code based on the review feedback and resubmit the update. This process may involve multiple rounds of iteration, so please remain responsive and communicative.
   The PR process will prompt the relevant Assignees. You can designate the relevant Assignees during the PR process, or contact us through the suggestions and communication channels in [README](../../../README_EN.md).

6. **Code Merging**

   A PR must collect the following four labels in sequence before the code can be merged:

   1. `ascend-cla/yes`: CLA check. You must sign the CLA when contributing for the first time. Once signed, this label is automatically obtained for each subsequent submission.
   2. `ci-pipeline-passed`: CI pipeline. Triggered by commenting `compile` in the PR process. If the CI pipeline check fails, you must revise the code based on the prompts and resubmit.
   3. `lgtm`: Provided by Reviewers. After Reviewers approve the changes, they comment `/lgtm` in the PR process to trigger the `lgtm` label.
   4. `approved`: Provided by Committers. After Committers approve the changes, they comment `/approved` in the PR process, which triggers the `approved` label.

   Once your PR has collected all four labels, your PR will be merged into the main branch.

#### PR Best Practices

- Keep PRs at a proper length for easier review
- One PR should address only one issue or implement one feature
- Respond to review comments in a timely manner
- Stay synchronized with the main branch and resolve conflicts promptly

#### Commit Message Specification

A commit message should clearly describe what was changed and why:

```text
<type>: <subject>

<body>

<footer>
```

The types include:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation updates
- `style`: Code formatting adjustments (no functional impact)
- `refactor`: Code refactoring
- `test`: Test-related changes
- `chore`: Changes to the build process or auxiliary tools

Example:

```text
feat: Add memory usage analysis feature

- Implement memory data collection module
- Add memory usage trend analysis algorithm
- Update related documentation

Closes #123
```

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and inclusive environment for all participants. By participating in this project, you agree to:

- Respect differing viewpoints and experiences
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Communication Channels

- **Issues**: Used for reporting bugs, proposing feature suggestions, and discussing technical issues
- **PRs**: Used for code review and discussing specific implementations
- **WeChat Group**: For daily communication and quick Q&amp;A (see [README](../../../README_EN.md) for suggestions and discussions)

## License

By contributing code to this project, you agree that your contributions will be licensed under the project's license. For details, see the [LICENSE](../legal/LICENSE) file.

The documentation under the docs directory of the msProbe tool is licensed under the CC-BY 4.0 license. For details, see [docs/LICENSE](../../LICENSE).

## Acknowledgments

Thank you for contributing to msProbe. Your efforts make this project more powerful and user-friendly. We look forward to your participation!

If you have any questions or need assistance, please feel free to ask in [Issues](https://gitcode.com/Ascend/msprobe/issues) or contact us through other community channels.
