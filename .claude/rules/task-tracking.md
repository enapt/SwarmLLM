# Task Tracking Rules

## Always create a task list for non-trivial work

When given a task (or set of tasks) that involves multiple steps, background agents, or any non-trivial work:

1. **Create tasks immediately** using `TaskCreate` before starting work
2. **Update task status** to `in_progress` when starting each task, `completed` when done
3. **Track parallel agent work** — create a collection/waiting task when launching background agents
4. **Break down large tasks** into discrete, actionable items

This ensures the user always has visibility into what's happening and progress is tracked.
