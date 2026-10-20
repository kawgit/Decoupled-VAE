# Project notes

## Type checking

Always check for and fix pyright errors and warnings after changing Python code. The project uses
pyright's `basic` mode, matching the editor setting, configured in `pyrightconfig.json`:

```bash
npx --yes pyright
```

The run must end with `0 errors, 0 warnings`. Fix the cause rather than adding `# type: ignore`.
