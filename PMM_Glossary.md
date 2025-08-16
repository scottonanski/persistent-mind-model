# 🧠 PMM Glossary (Plain English)

## Cross-Session Memory
- **What it means:** The AI doesn’t “forget” between runs — conversations and events are saved in a tamper-evident log.
- **Plain English:** Think of it as the AI keeping a diary that survives restarts.

## Personality Traits (Big Five)
- **What it means:** The AI has scores for openness, conscientiousness, extraversion, agreeableness, and neuroticism. These scores drift over time based on behavior.
- **Plain English:** The AI has a “personality” that slowly changes as it experiences things.

## Commitment Tracking
- **What it means:** If the AI says “Next, I will do X,” it’s logged as an open commitment. When evidence appears that it finished, it’s marked closed. Commitments are stored with SHA-256 hashes for integrity.
- **Plain English:** The AI keeps a to-do list of promises it makes and marks them done when finished.

## Reflection
- **What it means:** Every few interactions, the AI runs a reflection cycle to generate insights (short self-observations). These insights can shift traits or spawn commitments.
- **Plain English:** The AI occasionally steps back, thinks about what just happened, and writes down a lesson.

## Personality Drift
- **What it means:** Traits are adjusted based on commitments, reflections, and behavior patterns (e.g., completing tasks might boost conscientiousness).
- **Plain English:** The AI’s personality slowly evolves, like a person growing over time.

## Emergence Metrics (IAS/GAS)
- **What it means:** Identity Adoption Score (IAS) and Growth Acceleration Score (GAS) measure how much the AI refers to itself and seeks novelty. Stages S0–S4 mark identity development.
- **Plain English:** The AI has “growth stages,” like levels in a game, showing how much it’s developing a sense of self.

## Hash-Chained Memory
- **What it means:** Every event is linked with a SHA-256 hash to the previous one, creating a tamper-evident chain of memory.
- **Plain English:** The AI’s diary is cryptographically locked so past entries can’t be silently changed.
