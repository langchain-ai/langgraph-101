---
name: twitter-post
description: Write a Twitter/X post or thread based on research findings or a given topic. Use this skill when asked to create tweets, X posts, or social media threads.
---

# Twitter/X Post Skill

## Single Tweet Format

- Maximum 280 characters
- Lead with the most compelling point
- Use numbers or data when possible
- End with a link placeholder or call-to-action
- 1-2 hashtags max (optional)

## Thread Format (for longer content)

- **Tweet 1**: Hook + preview of what's coming (e.g., "A thread on X:" or "Here's what I found:")
- **Tweets 2-N**: One idea per tweet, numbered (1/, 2/, 3/)
- **Final tweet**: Summary + call-to-action + link
- Keep each tweet self-contained (people share individual tweets)
- 4-8 tweets is the sweet spot for engagement

## Tone

- Concise and punchy
- Opinionated takes perform better than neutral summaries
- Use plain language -- no corporate speak
- Contrarian or surprising angles get more engagement

## Tips

- Front-load the value (no throat-clearing or preambles)
- Use line breaks within tweets for readability
- Avoid hashtags in threads (they look spammy) -- save them for single tweets
- Numbers and lists catch the eye in a feed

## Example Single Tweet

```
AI agents that manage their context window well outperform those with 10x more tools.

The secret isn't more capabilities -- it's smarter context engineering.
```

## Example Thread

```
Thread: What makes AI agents actually work in production? 🧵

1/ It's not the model size. It's context management.

The best agents treat their context window like RAM -- offloading to filesystem, summarizing aggressively, loading info on demand.

2/ Subagents are the key to scaling.

Instead of one agent doing everything, delegate to specialists. The main agent only sees the summary, not 50 intermediate tool calls.

3/ Skills > giant system prompts.

Progressive disclosure: load detailed instructions only when the task needs them. Your agent's prompt stays clean until it matters.

4/ Memory needs structure.

Semantic (facts), episodic (experiences), procedural (rules) -- route them to different backends so they persist appropriately.

5/ The takeaway: the best agent architectures are about information flow, not raw capability.

What patterns are you using? Reply with your favorite agent architecture trick.
```
