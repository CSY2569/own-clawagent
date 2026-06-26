# Prompts Directory

## Structure

```
agents/<agent_id>/identity.md  — Layer 1: Agent identity (required, fallback if missing)
agents/<agent_id>/soul.md      — Layer 2: Personality and tone (optional)
shared/bootstrap.md            — Layer 3: Workspace and project context
shared/agents.md               — Layer 3: Multi-agent directory and routing info
```

## Adding a new agent

1. Create `agents/<new_id>/` directory
2. Create `identity.md` (required) and `soul.md` (optional)
3. Update `shared/agents.md` to describe the new agent
4. Set `CLAWAGENT_AGENT_ID=<new_id>` in `.env`

The `PromptBuilder` will automatically switch to the new agent's prompt files.
