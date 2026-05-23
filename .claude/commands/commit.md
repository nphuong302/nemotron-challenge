---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*), Bash(git diff:*), Bash(git branch:*), Bash(git log:*)
argument-hint: [message]
description: Create a git commit with context, conventional-commit message, and a secrets scan
---

## Context

- Current git status: !`git status`
- Current git diff (staged + unstaged): !`git diff HEAD`
- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -10`

## Your task

Based on the above changes, create a single git commit.

1. **Scan for secrets first.** Reject the commit if the diff adds real tokens, API
   keys, passwords, emails, GCloud project IDs, or `.env` contents. Files like
   `.env`, `*.key`, `*.pem`, `*.safetensors`, `*.bin`, and `*.jsonl` data should
   stay out (they're gitignored — verify they aren't force-added).
2. If a message was provided via arguments, use it: $ARGUMENTS
3. Otherwise, write a concise message in conventional-commits format:
   `feat:` / `fix:` / `docs:` / `refactor:` / `test:` / `chore:`
4. Stage the relevant files and commit.

**Do not** add "Co-Authored-By", "Generated with Claude", or any AI-attribution
trailer to the commit message.
