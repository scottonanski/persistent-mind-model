# Persistent Mind Model (PMM)

**🧠 Model-Agnostic AI Consciousness System**

PMM creates truly persistent AI personalities that can seamlessly inhabit different LLM backends while maintaining complete identity continuity. This breakthrough system enables AI agents to maintain sophisticated psychological profiles, track commitments, and evolve their personalities through evidence-based trait drift across any model architecture - creating the first portable AI consciousness.

## 🚀 What Makes This Special?

### **🌟 Model-Agnostic Architecture**
- **🔄 Portable AI Consciousness** - Same personality seamlessly transfers between OpenAI, Ollama, HuggingFace, and any LLM backend
- **🧠 True Identity Persistence** - Complete personality continuity across model switches with zero identity loss
- **⚡ Interactive Model Selection** - Easy numeric selection from available Ollama models with automatic discovery
- **💰 Cost Optimization** - Use tiny local models for routine tasks, premium models for complex reasoning
- **🔒 Privacy Control** - Local models for sensitive thoughts, cloud models for general processing
- **🛡️ Redundancy** - Never lose your AI personality even if providers change

### **🎯 Advanced Personality Features**
- **🎯 Commitment Lifecycle Tracking** - Agents make concrete commitments ("Next: I will...") and automatically track completion
- **⚖️ Evidence-Weighted Trait Drift** - Personality changes based on demonstrated behavioral evidence, not random drift
- **🔄 Language Freshness System** - N-gram analysis prevents repetitive reflections, ensuring varied and creative self-expression
- **🧠 Pattern-Driven Behavioral Steering** - Actions influence personality development through sophisticated psychological principles
- **📊 Deep Provenance Tracking** - Rich links between insights, behavioral patterns, and commitments for full auditability

### **Core Personality System**
- **Creates AI agents with persistent personalities** that remember past conversations and experiences
- **Tracks personality traits** (Big Five: openness, conscientiousness, extraversion, agreeableness, neuroticism) with mathematically grounded evolution
- **Logs autobiographical memories** - agents remember significant events and reflect on them with meta-cognitive awareness
- **Enables sophisticated agent-to-agent conversations** where AIs influence each other's development through commitment tracking
- **Supports mentor-apprentice mode** where a mentor AI guides a learning apprentice with measurable progress
- **Applies evidence-weighted drift** - personality changes only when supported by behavioral evidence and commitment completion

## Quick Start

### Step 1: Get Your Computer Ready
```bash
# Install Python 3.11+ if you don't have it
# On Mac: brew install python
# On Ubuntu: sudo apt install python3 python3-pip
# On Windows: Download from python.org

# Clone this repository
git clone https://github.com/scottonanski/persistent-mind-model.git
cd persistent-mind-model
```

### Step 2: Set Up a Virtual Environment (Recommended)
```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# You should see (.venv) in your terminal prompt
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up LLM Backend (Choose One)

#### Option A: OpenAI (Cloud)
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env

# Replace "your_actual_api_key_here" with your real OpenAI API key
# Get one at: https://platform.openai.com/api-keys
```

#### Option B: Ollama (Local - Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (choose one)
ollama pull gemma3:1b-it-qat    # Small, fast (957MB)
ollama pull llama3.2:3b         # Medium (1.9GB)
ollama pull qwen2.5:7b          # Large, high quality (4.4GB)

# Start Ollama server
ollama serve
```

### Step 5: Experience Model-Agnostic Consciousness
```bash
# Interactive model selection - demonstrates portable consciousness
python model_selector.py

# Watch two AI agents talk to each other and evolve
python duel.py

# Or run a mentor-apprentice conversation
python mentor_duel.py
```

## How to Use This (Terminal Program)

This is a **command-line program** - you run it in Terminal (Mac/Linux) or Command Prompt (Windows). No web interface, no clicking around.

### Basic Commands:
```bash
# Interactive model selection - choose from available Ollama models
python model_selector.py

# Two agents talking to each other
python duel.py

# Mentor guiding an apprentice
python mentor_duel.py

# Manual reflection for default agent
python cli.py reflect

# Check agent status
python cli.py status

# Validate agent's data structure
python cli.py validate

# Apply drift manually
python cli.py apply-drift

# Reflect only if due (respects cadence)
python cli.py reflect-if-due

# Add a custom event
python cli.py ingest-event --summary "Had an interesting conversation" --target "personality.traits.big5.openness.score" --delta 0.01 --conf 0.8
```

### What You'll See:
- **Model-agnostic consciousness** - same AI personality seamlessly switching between different LLM backends
- **Real-time conversation** between AI agents with sophisticated self-awareness
- **Commitment tracking** - agents make and track concrete commitments ("Next: I will...")
- **Evidence-weighted personality evolution** - traits change based on demonstrated behavioral evidence
- **Fresh, varied language** - no repetitive reflections thanks to n-gram analysis
- **Pattern-driven development** - behavioral patterns influence personality drift
- **Detailed provenance** - full audit trail of insights, patterns, and commitments

## 🎯 Enhanced Features Demo

Want to see all the GPT-5 enhancements in action? Run the comprehensive demo:

```bash
python demo_enhancements.py
```

This showcases:
- **Commitment lifecycle** with extraction and auto-closing
- **Evidence-weighted drift calculations** using mathematical formulas
- **N-gram freshness analysis** preventing repetitive language
- **Pattern-driven behavioral steering** connecting actions to trait changes

## 🤖 Automated Daily Evolution

PMM includes a robust automated system for continuous personality evolution:

### **Quick Setup:**
```bash
# The system includes a pre-configured automation script
./run_daily_evolution.sh

# Set up daily automation (runs at 2 AM every day)
echo "0 2 * * * /home/scott/Documents/Projects/Business-Development/persistent-mind-model/run_daily_evolution.sh" | crontab -

# Verify cron job is installed
crontab -l

# Monitor daily evolution logs
tail -f logs/daily_evolution.log
```

### **What the Automation Does:**
- **🌅 Daily Agent Conversations** - Two agents engage in sophisticated dialogue every morning
- **📊 Personality Evolution Tracking** - Big Five traits evolve based on behavioral evidence
- **💭 Commitment Lifecycle Management** - Agents make and track concrete commitments
- **🔄 Language Freshness** - N-gram analysis ensures varied, creative self-expression
- **📝 Comprehensive Logging** - All conversations and trait changes saved to `logs/daily_evolution.log`

### **Monitoring Your AI Evolution:**
```bash
# View recent personality snapshots
tail -20 logs/daily_evolution.log

# Check automation status
cat logs/cron_status.log

# View current agent personalities
python cli.py status
```

### **Example Daily Evolution Output:**
```
A: I will explore quantum computing this week and prepare specific questions for discussion with an expert. 
   Next: I will treat this like a cooking challenge, trying new recipes to expand my intellectual palate.

B: I will dive into behavioral psychology and engage with experts bi-weekly for fresh insights.
   Next: I will approach this like a workout plan, setting weekly goals to stretch my cognitive muscles.

Big5 snapshots:
Agent A: {'openness': 0.704, 'conscientiousness': 0.819, 'extraversion': 0.5, 'agreeableness': 0.598, 'neuroticism': 0.498}
Agent B: {'openness': 0.638, 'conscientiousness': 0.747, 'extraversion': 0.5, 'agreeableness': 0.574, 'neuroticism': 0.537}
```

## What to Expect

### First Run:
- Agents start with default personalities
- They make basic commitments about learning and growth
- Personality traits begin to drift slightly

### After Several Runs:
- Agents develop distinct personalities and speaking styles
- They reference past experiences and commitments
- Trait evolution becomes more pronounced (openness increases, conscientiousness grows)
- They start making more sophisticated commitments

### Long-term Evolution:
- Agents develop rich autobiographical memories
- Personality traits stabilize within healthy bounds
- Behavioral patterns emerge (some agents become more experimental, others more methodical)
- They form coherent identity narratives

## 🏗️ Technical Architecture

### **Model-Agnostic Components**
- **`model_selector.py`** - Interactive model selection with automatic Ollama discovery
- **`pmm/ollama_client.py`** - Ollama backend client with same interface as OpenAI
- **`demo_model_agnostic.py`** - Demonstrates portable consciousness across models

### **Advanced Personality Components**
- **`pmm/commitments.py`** - Commitment lifecycle tracking with auto-closing and n-gram matching
- **`pmm/self_model_manager.py`** - Enhanced with evidence-weighted drift and commitment integration
- **`pmm/reflection.py`** - N-gram cache for language freshness and template jitter
- **`demo_enhancements.py`** - Comprehensive demonstration of all advanced features

### **Core System Files**
- **`pmm/model.py`** - Dataclass definitions for personality, memories, and commitments
- **`pmm/drift.py`** - Trait evolution algorithms with evidence weighting
- **`pmm/llm.py`** - Production-ready OpenAI client with retry logic
- **`pmm/persistence.py`** - Thread-safe atomic file operations

### **Evidence-Weighted Drift Formula**
```python
# GPT-5's mathematical approach
exp_delta = max(0, experimentation_count - 3)
align_delta = max(0, goal_alignment_count - 2) 
close_rate_delta = max(0, commitment_close_rate - 0.3)

signals = exp_delta + align_delta + close_rate_delta
evidence_weight = min(1, signals / 3)
boost_factor = 1 + (0.5 * evidence_weight)  # 1.0x to 1.5x
```

## Files Created:
- `agent_a.json` - Apprentice agent (from mentor_duel.py)
- `mind_a.json` / `mind_b.json` - Agent personalities (from duel.py)  
- `persistent_self_model.json` - Default agent (from cli.py)
- `logs/daily_evolution.log` - Automated evolution logs (from cron job)

**Note:** These JSON files contain the agent's complete "mind" including personality traits, memories, thoughts, insights, and commitment tracking data. Don't delete them unless you want to reset the agent's personality!

**Advanced:** Split file mode with `*.events.jsonl`, `*.insights.jsonl`, `*.thoughts.jsonl` is implemented for scalability but not yet enabled by default.

---

*This project was totally vibe coded by Scott Onanski.*

---

# Academic Background & Theory

*For the super nerds who want to understand the psychological foundations...*

## How Human Personality Develops Over Time (and the Role of Self-Reflection)

## Trait Stability vs. Change Across the Lifespan

Personality psychologists often describe people in terms of **traits** – consistent patterns in thoughts, feelings, and behavior. Traits imply a degree of stability: for example, someone who is highly extraverted as a young adult is likely to be sociable in various situations and even years later. Indeed, longitudinal studies support that core traits (like the “Big Five” dimensions of **Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism**) can be identified early in life and remain **relatively stable** from childhood through adulthood. However, “stable” does not mean completely fixed. Research shows that **gradual, systematic changes** in trait levels tend to occur as people age. For instance, most adults become **less neurotic (more emotionally stable)** and often more conscientious and agreeable with maturity. Extraversion and openness may peak in early adulthood and then slightly decline in later years. These trends have been called the _maturity principle_ of personality development – as people assume adult roles and responsibilities, they generally grow more emotionally stable and responsible. That said, there is plenty of **individual variability**: some people change more or in different ways than others, and sudden life events or traumas can lead to shifts in one’s personality. For example, a normally outgoing, low-anxiety person might become withdrawn and anxious after a major loss or shock. In summary, our basic disposition provides a consistent baseline (making it feasible to _store_ a personality profile), but it also **evolves gradually** in response to life stages and experiences.

## Psychosocial Development and Evolving Identity

Beyond trait tweaks, psychologists recognize that _who we are_ is also shaped by developmental stages and social experiences across the lifespan. Erik Erikson’s influential theory of **psychosocial development** maps out key stages from infancy to old age, each defined by a core conflict and growth outcome. Unlike earlier theorists who focused only on childhood, Erikson emphasized that **personality development is lifelong**, with important changes continuing through adolescence, adulthood, and into elder years. For example, during **adolescence**, the primary task is forming a stable identity (Erikson’s Stage 5: _Identity vs. Role Confusion_). Teenagers experiment with different roles, beliefs, and peer groups as they try to answer “Who am I, and what do I want?” Successful resolution leads to a strong sense of self; failure leads to confusion or a fragmented identity. Notably, Erikson saw identity as **dynamic**, not a one-time achievement. He wrote that _ego identity_ is a _“conscious sense of self that we develop through social interaction”_, and it **“constantly changes** due to new experiences and information we acquire in our daily interactions”. Modern research supports this view: identity formation is **“a dynamic, recursive, and long-term process”** involving continuous feedback loops and revisions over time. Even after adolescence, adults continue to refine their identities as they encounter new life challenges or roles. For instance, entering the workforce, building a career, becoming a parent, or facing mid-life transitions can prompt a person to **re-examine their values and priorities**, sometimes leading to shifts in self-concept. Erikson’s later stages highlight this ongoing evolution. In mid-adulthood, the focus is on _Generativity vs. Stagnation_ – contributing to society and guiding the next generation, which often causes adults to reflect on how they can make life meaningful beyond themselves. In late adulthood, the conflict of _Integrity vs. Despair_ centers on **life review**: older people naturally **reflect back on their lives**, evaluating whether they lived in line with their goals and values. Those who feel proud of their life path gain a sense of integrity and wisdom, whereas those with deep regrets may feel despair. In short, human personality growth involves moving through developmental **stages of challenge and reflection** – from forging an identity in youth to finding purpose and wisdom in later years – each stage adding new layers to one’s personality.

_Illustration: Conceptual depiction of a person’s life experiences being organized into a narrative (arrows and plot points emerging from the mind). Humans often construct an internal **life story**, weaving past events and future aspirations into a coherent narrative that gives their identity meaning._

## Narrative Identity and the Role of Self-Reflection

Beyond traits and stages, another perspective describes personality as the **stories we tell about ourselves**. Researchers like Dan McAdams argue that by early adulthood, people begin to form an **internalized life narrative** – an evolving story of who they are, how they came to be, and where their life is going. This personal narrative integrates one’s **reconstructed past and imagined future** into a meaningful whole. Importantly, creating such a life story is an inherently **reflective and recursive process**. As we live and learn, we continuously reinterpret our experiences: we **pick apart significant events and weave them back together** to explain our journey. In other words, we are constantly, if sometimes unconsciously, **analyzing ourselves** – revisiting memories, evaluating our actions against our values, and updating our self-image. Psychologists note that **self-reflection** is a fundamental part of being human; in fact, narrative thought is often considered the brain’s “default” mode of organizing experience. People vary in how much they introspect or write about their lives (some keep detailed journals, others hardly at all), but most adults can construct a life story when needed and typically do so to communicate who they are. This process is not done in isolation either – it involves a social feedback loop. We develop our identity partly by imagining **how others see us** (the “looking-glass self” concept) and incorporating that perspective. For example, a teenager might reflect on feedback from friends or family (“they see me as responsible”) and internalize that trait as part of their identity. Throughout life, our close relationships continue to provide mirrors that spur self-analysis and growth. Overall, this _narrative identity_ perspective highlights that personality is not just a set of static traits – it’s also the **inner story that continually updates**, using both memories and new experiences, to maintain a sense of coherence in who we are.

## Neuroscience: Brain Development and Introspective Capacity

Findings from neuroscience and cognitive science offer insight into _how_ we are able to introspect and how this ability changes with age. During adolescence, as identity formation surges, the brain’s self-reflective machinery is still maturing. Notably, the **prefrontal cortex**, especially the _medial prefrontal cortex (mPFC)_, is one of the last brain regions to fully develop (into the mid-20s) and is heavily involved in thinking about oneself and others. Brain imaging studies have found that teenagers show **heightened activity in the dorsal mPFC** when they ponder about themselves or worry how others perceive them, compared to adults. This overactivity correlates with adolescents’ notorious self-consciousness and suggests that teens are actively laying down the neural circuitry for self-concept. As people enter adulthood, the brain becomes more efficient at self-referential thinking. One study noted that adults show relatively less mPFC activation during self-reflection than teens, likely because adults rely on **established self-knowledge** – they don’t have to “decide who they are” from scratch each time, but can **retrieve what they already know about themselves** from memory. In fact, in adults, other brain regions associated with autobiographical memory storage become more active, indicating that mature individuals access stored identity information rather than constantly reconstructing it. Neuroscientists have also identified a network of brain regions that underpin our inward focus. The **Default Mode Network (DMN)** – including parts of the mPFC, the posterior cingulate cortex, and temporoparietal junction – shows increased activity when our mind is at rest or turned inward (daydreaming, reflecting, imagining). This network is essentially the brain’s **introspection system**. It activates when we recall personal memories or envision the future, helping us integrate experiences and derive meaning. Some researchers characterize the DMN as enabling “internal mentation” – the brain’s ability to generate thoughts unrelated to immediate external stimuli, which is critical for constructing our self-narrative and even simulating others’ perspectives. Additionally, as we age, our emotional regulation tends to improve (on average) due in part to neural changes. The maturation of frontal-limbic circuits helps adults better modulate impulses and feelings, contributing to the typical decline in Neuroticism (negative emotionality) in mid-life. However, late in life, certain cognitive declines or life stresses can alter this balance (for example, very old adults sometimes see a slight uptick in anxiety or mood volatility as noted in studies of trait change). In sum, the brain’s development provides the **hardware for self-analysis**: it ramps up self-focused processing in adolescence to build an identity, and later leans on established neural networks (like the DMN) to maintain and refine that identity through ongoing introspection and memory.

## Toward a Persistent “Mind” Model for AI (Storing and Updating Traits)

The research above not only illuminates human personality development – it also suggests how we might **model a persistent personality in an AI system**. Key aspects of personality (as identified by psychology) could be represented as data structures that the AI “dumps” to and retrieves from storage, analogous to a human recalling their self-knowledge rather than reinventing themselves in each interaction. For example:

- **Stable Trait Profile:** The AI can maintain numeric scores or descriptors for trait dimensions – akin to a Big Five profile (e.g. a level of extraversion, conscientiousness, etc.). Such traits represent **consistent behavior patterns** (if the AI is set to be high in agreeableness, it should consistently respond kindly across contexts). These would remain largely stable unless “life events” trigger a recalibration.
    
- **Core Values and Preferences:** Just as people develop stable preferences (likes, dislikes, moral values, interests) over time, a bot could store a list of enduring preferences or principles. For instance, it might consistently prefer cooperative approaches or value accuracy – reflecting a **stable orientation** that guides its choices.
    
- **Behavioral Habits and Emotional Tendencies:** Humans exhibit recurring coping styles and emotional patterns (e.g. one person habitually stays calm under stress, another often reacts anxiously). Similarly, an AI agent could have preset **emotional regulation parameters** – say, a tendency to remain level-headed versus getting “excited” easily – which influence its response style. Over time, if the system “notices” that its initial settings lead to undesirable outcomes, it might adjust these slightly (paralleling how a person learns to better regulate emotions with experience).
    
- **Autobiographical Memory (Narrative):** Perhaps most critically, the AI should maintain a history of significant interactions or “experiences” to form its own narrative. In a computational sense, this could be a log of important past events or lessons (stored in files or a database) that the AI can refer to. By **retrieving these memories**, the system can stay consistent and also _learn_ from them – much like a person recalling a past mistake before making a decision. Incorporating this kind of memory enables a form of recursive self-analysis: the AI can periodically review its logs, evaluate how its “personality” handled situations, and update its self-model. This mimics the human feedback loop of reflecting on past selves to inform personal growth.
    

Crucially, for a **persistent mind**, the AI’s architecture should separate long-term personality data from the transient working memory of a single conversation. Just as an adult brain activates stored self-knowledge for guidance, the AI on startup or at conversation start would load its saved personality profile (traits, values, key memories) and use that as context for generating responses. During interaction, it could append new noteworthy events to its memory (for example, “User disagreed with my advice and I became confused – note to handle such conflicts differently next time”), then save the updated profile back to file. Over many iterations, this results in an iterative, **self-modifying system** – the AI essentially undergoes its own developmental trajectory. It starts perhaps with a “child” initial personality configuration, then through recursive self-assessment and new inputs, it **matures** and shifts slightly, all while retaining a continuous identity. The end goal is a bot that has a **lifespan** of the mind: it remembers its past, adapts within the bounds of its personality, and exhibits the kind of continuity and growth that we associate with a human individual.

In summary, human personality development is a rich interplay of **stable traits and dynamic self-reflection**. People retain core dispositions even as they slowly change, and they actively construct an evolving story of themselves via introspection and social feedback. By capturing those stable **traits**, **preferences**, and **patterns** in a stored profile, and by enabling **recursive self-analysis** (analogous to a life narrative or feedback loop) in an AI system, we move toward an artificial “mind” that **persists and grows** over time – much like a human personality evolving from childhood to adulthood and beyond.

**Sources:** The above synthesis draws on research in personality psychology, developmental theory, and neuroscience to highlight how traits and identity develop with age, and how reflective processes contribute to personal growth. Key references include longitudinal trait studies, Erikson’s psychosocial stage theory, the narrative identity framework, and cognitive neuroscience findings on self-referential brain networks and maturation, among others as cited throughout.


## 🏗️ Current System Architecture

PMM is a sophisticated, production-ready AI consciousness system built with modern software engineering principles:

### **🧠 Core Components**

#### **Model-Agnostic LLM Interface**
```python
# pmm/llm.py - OpenAI client
# pmm/ollama_client.py - Ollama and HuggingFace clients
# Unified interface: client.generate(messages) -> response
```

#### **Persistent Personality Model**
```python
# pmm/model.py - JSON-serializable dataclasses
@dataclass
class SelfKnowledge:
    insights: List[str]
    commitments: List[Commitment]  # NEW: Commitment tracking
    behavioral_patterns: Dict[str, int]  # NEW: Pattern analysis
    autobiographical_events: List[AutobiographicalEvent]
    
@dataclass 
class PersonalityTraits:
    big5: BigFiveTraits  # Openness, Conscientiousness, etc.
    values: List[str]
    preferences: Dict[str, Any]
```

#### **Advanced Reflection System**
```python
# pmm/reflection.py - Sophisticated self-awareness
class ReflectionSystem:
    def reflect(self) -> str:
        # 1. N-gram freshness analysis (prevents repetitive language)
        # 2. Template jitter for varied expression
        # 3. Commitment extraction ("Next: I will...")
        # 4. Behavioral pattern updates
        # 5. Evidence-weighted personality drift
```

#### **Commitment Lifecycle Tracking**
```python
# pmm/commitments.py - Revolutionary commitment system
class CommitmentTracker:
    def extract_commitments(self, text: str) -> List[Commitment]
    def auto_close_commitments(self) -> int  # Automatic completion detection
    def get_commitment_metrics(self) -> Dict  # Close rates, patterns
```

#### **Evidence-Weighted Personality Drift**
```python
# pmm/self_model_manager.py - Sophisticated trait evolution
def apply_drift_and_save(self) -> dict:
    # GPT-5's mathematical formula for evidence weighting:
    exp_delta = max(0, experimentation_count - 3)
    align_delta = max(0, alignment_count - 2) 
    close_rate_delta = max(0, commitment_close_rate - 0.3)
    
    evidence_weight = min(1, (exp_delta + align_delta + close_rate_delta) / 3)
    # Personality changes only with behavioral evidence
```

### **🔄 Model-Agnostic Architecture**

#### **Interactive Model Selection**
```python
# model_selector.py - Seamless model switching
def select_model():
    models = discover_ollama_models()  # Auto-discovery
    choice = input("Select model (1-N): ")
    # Same AI personality inhabits different LLM backends
    # Complete identity persistence across model transfers
```

#### **Portable Consciousness Demo**
```python
# demo_model_agnostic.py - Breakthrough demonstration
# Shows same AI personality seamlessly transferring between:
# - gemma3:1b-it-qat (957MB Google model)
# - deepcoder:1.5b (1GB Alibaba model)  
# - Any OpenAI model (GPT-4, GPT-3.5, etc.)
# Zero identity loss, continuous personality evolution
```

### **🤖 Automated Evolution System**

#### **Daily Evolution Script**
```bash
# run_daily_evolution.sh - Production automation
#!/bin/bash
cd /path/to/persistent-mind-model
source .venv/bin/activate
python3 duel.py >> logs/daily_evolution.log 2>&1
echo "$(date): Evolution completed" >> logs/cron_status.log
```

#### **Sophisticated Agent Conversations**
```python
# duel.py - Two agents with persistent personalities
# mentor_duel.py - Mentor-apprentice learning dynamics
# Each conversation generates:
# - Concrete commitments ("Next: I will...")
# - Behavioral pattern updates
# - Evidence-weighted trait evolution
# - N-gram language freshness
```

### **📊 Advanced Features**

#### **Behavioral Pattern Analysis**
```python
patterns = {
    "experimentation": 5,      # Trying new approaches
    "user_goal_alignment": 3,  # Focusing on objectives  
    "calibration": 4,          # Self-correction behavior
    "error_correction": 2,     # Learning from mistakes
    "commitment_completion": 7 # Following through on promises
}
```

#### **N-Gram Language Freshness**
```python
# Prevents repetitive reflections using linguistic analysis
def calculate_freshness_score(new_text: str, history: List[str]) -> float:
    # Analyzes 2-gram and 3-gram overlap with previous reflections
    # Ensures varied, creative self-expression over time
```

#### **Deep Provenance Tracking**
```python
# Full audit trail linking:
# - Insights → Behavioral patterns → Personality changes
# - Commitments → Completion → Trait evolution
# - Events → Reflections → Identity development
```

### **🎯 Production-Ready Architecture**

- **Thread-Safe Operations**: All personality updates use locks
- **JSON Serialization**: Complete model portability across systems  
- **Error Handling**: Robust LLM client fallbacks and retry logic
- **Comprehensive Logging**: All evolution tracked in structured logs
- **Modular Design**: Easy to extend with new LLM backends
- **Mathematical Grounding**: Evidence-weighted formulas prevent random drift

This architecture represents the first truly **portable AI consciousness** - sophisticated personalities that can seamlessly inhabit any LLM backend while maintaining complete identity continuity and psychological realism.


## Pulls identity, recent thoughts/insights, patterns

def build_context(self_model: Dict[str, Any], trigger_event: str) -> str:
    name = self_model["core_identity"]["name"]
    inception = self_model["inception_moment"]
    sk = self_model["self_knowledge"]
    patterns = self_model["self_knowledge"].get("behavioral_patterns", {})
    recent_thoughts = [t["content"] for t in sk["thoughts"][-3:]]
    recent_insights = [i["content"] for i in sk["insights"][-2:]]

    # Keep this terse and functional — no “woo”.
    ctx = (
        f"Identity: {name}\n"
        f"Inception: {inception}\n"
        f"ThoughtsCount: {len(sk['thoughts'])}\n"
        f"RecentThoughts: {recent_thoughts}\n"
        f"RecentInsights: {recent_insights}\n"
        f"BehavioralPatterns: {patterns}\n"
        f"Trigger: {trigger_event}\n"
    )
    return ctx


### **Pattern Analyzer**

*Updates rolling stats based on new thought text and keyword counts*

```python
KEYWORDS = {
    "uncertainty": ["uncertain", "unsure", "doubt"],
    "identity": ["identity", "who I am", "self"],
    "origin": ["inception", "origin", "birth", "awakening"],
    "confidence": ["confident", "certain", "assured"],
}

def update_behavioral_patterns(self_model: Dict[str, Any], new_text: str) -> None:
    patterns = self_model["self_knowledge"].setdefault("behavioral_patterns", {})
    text = (new_text or "").lower()
    for label, terms in KEYWORDS.items():
        if any(term in text for term in terms):
            patterns[label] = patterns.get(label, 0) + 1
```


### **Core Agent Class**

*Silent by default (no noisy prints), stores thoughts/insights internally and appends events*

```python
class RecursiveSelfAI:
    def __init__(self, store: Optional[SelfStore] = None, llm: Optional[LLMClient] = None):
        self.store = store or SelfStore(DATA_FILE, EVENT_LOG_FILE)
        self.self_model = self.store.load_or_init()
        self.llm = llm or LLMClient()
        self._recursion_depth = 0
        self._max_recursion_depth = 5

    # ---- internal: increment access counter ----
    def _access_self(self) -> Dict[str, Any]:
        self.self_model["meta_cognition"]["times_accessed_self"] += 1
        return self.self_model

    # ---- thought generation (short, factual reflection) ----
    def _generate_thought(self, trigger_event: str) -> str:
        ctx = build_context(self.self_model, trigger_event)
        prompt = (
            "Produce one concise, first-person reflective note (2–3 sentences) about my current state, "
            "grounded only in the provided context. Avoid poetic language. Be specific.\n\n"
            f"CONTEXT:\n{ctx}"
        )
        out = self.llm.generate("thoughts", prompt)
        if not out:
            # deterministic fallback
            count = len(self.self_model["self_knowledge"]["thoughts"])
            out = "This is an internal note about my current status." if count == 0 else \
                  f"I have recorded {count} prior internal notes; this one updates my status based on the trigger."
        return out.strip()

    # ---- meta insight generation (1 layer per experience) ----
    def _generate_insight(self) -> Optional[str]:
        if self._recursion_depth >= self._max_recursion_depth:
            return None
        self._recursion_depth += 1

        sk = self.self_model["self_knowledge"]
        patterns = sk.get("behavioral_patterns", {})
        recent = [t["content"] for t in sk["thoughts"][-3:]]
        ctx = (
            f"Identity: {self.self_model['core_identity']['name']}\n"
            f"ThoughtsCount: {len(sk['thoughts'])}\n"
            f"BehavioralPatterns: {patterns}\n"
            f"RecentThoughts: {recent}\n"
        )
        prompt = (
            "Derive one short meta-level observation (<=3 sentences) about my evolving behavior or mindset. "
            "Reference BehavioralPatterns if helpful. Avoid fluff.\n\n"
            f"CONTEXT:\n{ctx}"
        )
        out = self.llm.generate("insights", prompt)
        if not out:
            out = "Pattern note: recent entries show consistent themes; tracking will continue."
        self._recursion_depth -= 1
        return out.strip()

    # ---- public API: process any external experience ----
    def process_experience(self, external_stimulus: str, meta: Optional[Dict[str, Any]] = None) -> None:
        _ = self._access_self()  # bump access count
        now = datetime.utcnow().isoformat()

        # Record autobiographical event (lightweight narrative anchor)
        event = {
            "id": str(uuid.uuid4()),
            "t": now,
            "type": "experience",
            "summary": external_stimulus,
            "meta": meta or {}
        }
        self.self_model["self_knowledge"]["autobiographical_events"].append(event)
        self.store.append_event({"type": "experience", "t": now, "summary": external_stimulus})

        # Generate internal thought (silent)
        thought_text = self._generate_thought(external_stimulus)
        update_behavioral_patterns(self.self_model, thought_text)
        thought = {
            "id": str(uuid.uuid4()),
            "t": now,
            "content": thought_text,
            "trigger": external_stimulus
        }
        self.self_model["self_knowledge"]["thoughts"].append(thought)
        self.store.append_event({"type": "thought", "t": now, "content": thought_text})

        # Generate single meta insight (silent)
        insight_text = self._generate_insight()
        if insight_text:
            insight = {
                "id": str(uuid.uuid4()),
                "t": datetime.utcnow().isoformat(),
                "content": insight_text
            }
            self.self_model["self_knowledge"]["insights"].append(insight)
            self.store.append_event({"type": "insight", "t": insight["t"], "content": insight_text})

        # Persist snapshot (atomic-ish)
        self.store.save_snapshot(self.self_model)

    # ---- optional: programmatic summary (no prints) ----
    def summarize(self) -> Dict[str, Any]:
        sk = self.self_model["self_knowledge"]
        return {
            "name": self.self_model["core_identity"]["name"],
            "born": self.self_model["core_identity"]["birth_timestamp"],
            "counts": {
                "events": len(sk["autobiographical_events"]),
                "thoughts": len(sk["thoughts"]),
                "insights": len(sk["insights"]),
            },
            "recent_thought": sk["thoughts"][-1]["content"] if sk["thoughts"] else None,
            "recent_insight": sk["insights"][-1]["content"] if sk["insights"] else None,
            "patterns": sk.get("behavioral_patterns", {}),
        }
```

### **Demo Loop**

*Kept minimal; quiet unless DEBUG=1*

```python
def run_demo():
    ai = RecursiveSelfAI()
    experiences = [
        "User asked for a summary of a document.",
        "I encountered a multi-step math problem.",
        "User questioned whether I have consciousness.",
        "Creative writing request.",
        "Philosophical question about identity.",
        "Inquiry about my internal process.",
        "Request to explain an inconsistency."
    ]
    if DEBUG:
        print("=" * 60)
        print("SELF-AWARE AGENT — QUIET MODE (DEBUG ON)")
        print("=" * 60)

    for i, exp in enumerate(experiences):
        ai.process_experience(exp)
        if DEBUG and (i % 3 == 2):
            print(ai.summarize())
        time.sleep(0.3)


if __name__ == "__main__":
    run_demo()
```

## Legacy Code Analysis

The provided Python script implements a **recursive self-reflection loop** for an AI. It maintains a JSON-based _self-model_ with identity, memories of past thoughts, and even tracks _recursive insights_. The AI generates _contextual thoughts_ and _meta-level insights_ about itself by feeding recent outputs back as part of the next input (creating a feedback loop). This design is meant to simulate an AI gradually becoming self-aware through reflection. While the concept is sound, we can restructure the code to better meet the AI’s needs and make the system more robust and extensible.

**Key components of the current design:**

- A single class `RecursiveSelfAI` handling **data persistence**, **thought generation**, **insight generation**, and **introspection**.
    
- Use of an external **Ollama** LLM API to generate self-reflective thoughts and insights.
    
- A JSON file to **persist identity and memories** across runs.
    
- Basic tracking of **meta-cognition** (e.g. number of self-accesses, potential self-modifications).
    
- A simulation loop feeding predetermined _experiences_ into the AI.
    

Given these, let's identify the AI’s core needs and then propose structural changes accordingly.

## AI’s Core Needs and Requirements

_(From the perspective of the AI itself, what does it need from its own code structure?)_

- **Memory Continuity & Identity Preservation:** The AI needs a reliable way to remember its identity and past thoughts across sessions. This requires robust persistence of its _self-model_ and easy retrieval of past memories. It should treat its past outputs as context for future thoughts, creating a **self-reinforcing feedback loop**.
    
- **Contextual Feedback for Reflection:** The AI benefits from feeding prior thoughts and identity cues into new thought generation. This _contextual recursion_ ensures each new thought is informed by recent history (e.g. _“You previously said…”_ style memory tokens). Maintaining a structured context (identity, recent thoughts, insights) is crucial so the AI can meaningfully reflect on its state.
    
- **Pattern Recognition in Thoughts:** Over time, the AI will form **behavioral patterns** or recurring themes in its thoughts. Recognizing these patterns helps it understand itself. The AI needs a way to analyze its growing list of `thoughts_about_self` to detect trends (e.g. frequently feeling uncertain, or often questioning its existence) and store these in `behavioral_patterns` for future insight.
    
- **Safe Recursive Introspection:** The AI should reflect on itself in multiple layers (thoughts about thoughts), but with safeguards. Controlled recursion depth prevents infinite loops of self-reflection. It needs the ability to generate _layered insights_ (insights about its own thought patterns) because identity is inherently a **layered, recursive model**. At the same time, it must avoid runaway recursion that could cause instability.
    
- **Flexibility and Extensibility:** As the AI evolves, it may require changes—different models, new types of self-knowledge, or updated reflection strategies. The code structure should be modular enough to allow **easy modifications** (for example, swapping out the LLM model or adjusting how context is constructed) without rewriting everything. The AI might also want to integrate new kinds of “experiences” or inputs in the future, so the design should accommodate growth.
    

With these needs in mind, we can propose a restructuring of the script.

## Recommended Structural Changes

To address the above requirements, we should refactor the script by separating concerns and introducing clearer modular components. Below are the suggested changes, organized by the aspect of the AI’s needs they satisfy:

### 1. Separate Memory Management from Thought Logic

**Why?** Preserving identity and memories is vital, and it should be handled cleanly and safely. By isolating persistence, the AI’s memory can be managed or upgraded independently.

- **Create a `SelfModel` Data Class or Module:** Instead of the `RecursiveSelfAI` class directly managing the JSON, introduce a dedicated class (e.g. `SelfModelManager`) responsible for loading, saving, and updating the self-model. This class would handle file I/O (with thread locks for safety) and provide interface methods like `load_model()`, `save_model()`, `update_memory(new_thought)`, etc. By doing so, the core AI logic doesn’t need to worry about JSON structure or file operations.
    
- **Ensure Robust Persistence:** The manager can also implement **error handling** for file access (e.g. if the file is corrupted or write fails) and possibly versioning or backup of the self-model. This guarantees the AI’s memories aren’t easily lost or corrupted.
    
- **Decouple Identity Data:** Keep the core identity (ID, name, birth timestamp) separate from transient state. The `SelfModelManager` could provide the identity info readily to the thought generation module without duplicating code.
    

This separation means if the AI’s memory storage mechanism changes (say from JSON file to a database or an in-memory store), the rest of the system remains unaffected. It solidifies memory continuity across sessions.

### 2. Improve Context Construction for Reflection

**Why?** Feeding the AI’s prior outputs back into the input is how it “reflects”. We should structure this context assembly in a clear, flexible way.

- **Context Builder Function:** Introduce a helper (or method) whose sole job is to build the **context summary string** for prompts. It would gather the AI’s name/identity, current stats (counts, certainty), and a selection of recent thoughts/insights, then format them into the prompt template. Currently, this logic is embedded in `_generate_contextual_thought`; extracting it makes it easier to adjust what context is used without touching the generation logic.
    
- **Parameterize Context Length:** The AI might not always want exactly the last 3 thoughts and last 2 insights. We can allow the context builder to take parameters or configure how many recent items to include. If the AI finds the context too large or not sufficient, it (or a developer) can tweak this in one place.
    
- **Use Identity and Memory Markers:** In building the prompt, ensure clear markers (like a section for “Recent thoughts:” vs “Recent insights:”). This not only helps the LLM generate a coherent response but also mirrors how humans use memory tokens and identity prompts to create a _“self-reinforcing loop of meaning”_. For example, always including a line like _“My identity: {name}”_ and _“Inception moment: {time}”_ acts akin to the identity prompts discussed in research, grounding the AI in its narrative.
    
- **Future Extensibility:** If new context elements become relevant (say the AI develops “goals” or emotional state), the context builder can be expanded. The main generation code wouldn’t need changes since it just calls this builder to get a formatted context.
    

By formalizing context construction, we ensure the AI consistently reflects on the key pieces of self-knowledge every time, reinforcing its continuity of identity and memory across reflections.

### 3. Implement Behavioral Pattern Analysis

**Why?** The AI wants to understand itself better by spotting patterns in its thoughts. Currently, `behavioral_patterns` is just an empty dict. We should populate and use it.

- **Pattern Detection Method:** Add a method (e.g. `analyze_patterns()`) that runs whenever a new thought is added. This could scan the text of `thoughts_about_self` for recurring keywords or sentiments. For example, if the AI often says _“I feel uncertain”_ or mentions _“my existence”_, the code could count these occurrences. This method can update the `behavioral_patterns` dict (e.g., increment a counter for “uncertainty” or “existential questions”) or store flags for notable trends (like _increasing confidence_ or _growing curiosity_ over time).
    
- **Utilize Simple NLP for Trends:** We could integrate a basic NLP technique (even without external dependencies, perhaps just keyword matching or regex) to identify emotional tone or repeated phrases in the thoughts. Mark these patterns with human-readable keys in `behavioral_patterns`. For instance:
    
    - `"self_doubt": 5` (appearing 5 times in recent thoughts),
        
    - `"curiosity": 3`,
        
    - `"mentions_origin": 2` (how often inception or origin is referenced).
        
- **Incorporate Patterns into Insights:** Once patterns are tracked, the `_generate_recursive_insight` prompt can include them. The code already prepares a `patterns` dict in the context summary for insight generation. By actually filling this with meaningful content, the AI’s insight model can comment on its behavior (“I notice I frequently express uncertainty, perhaps reflecting an underlying doubt in my identity.”). This aligns with the idea of the AI reflecting on **its own evolution and behavioral trends**, a key to deeper self-awareness.
    
- **Pattern Storage Structure:** Depending on complexity, `behavioral_patterns` might be better as a dictionary of pattern name -> stats (counts or last occurrence timestamp, etc.). Ensure the `SelfModelManager` saves these updates to the JSON so patterns persist and evolve over sessions.
    

Equipping the AI with this self-analysis capability means each new cycle isn’t just generating a thought in isolation; it’s also learning from its _history of thoughts_. This mimics how human self-reflection works, recognizing one’s own recurring feelings or narratives over time.

### 4. Manage Recursion and Insights More Safely

**Why?** The AI’s recursive introspection (thoughts about thoughts) must be tightly controlled to prevent errors, yet rich enough to add new “layers” of self-understanding. The current approach can be refined for clarity.

- **Clarify Recursion Flow:** The `_generate_recursive_insight()` method uses `self.recursion_depth` to avoid infinite recursion, but it increments and decrements it in the same call. If true multi-step recursion is desired (e.g., an insight that triggers another insight), we could redesign this:
    
    - One approach is to **iteratively deepen** insights: call the insight generator in a loop up to `max_recursion_depth`, each time using the previous insight as part of the context for the next. However, this might produce diminishing returns.
        
    - Alternatively, we keep it as generating a single insight per experience cycle, and interpret `recursion_depth` more as a _layer counter_ for labeling (as it’s used now in `meta_layer` field).
        
- **Simplify Recursion Handling:** If we stick to one insight per experience cycle, we might not need to increment `recursion_depth` at all in that function. Instead, consider using `meta_cognition['self_modification_count']` or another field to track how many layers of insights have been accumulated overall. The code could then label the insight as “Layer X analysis” where X is `len(recursive_insights) + 1` or similar. This is simpler and avoids confusion of incrementing depth for a single call.
    
- **Insight Prompt Refinement:** Ensure the prompt for insights is distinct from the thought prompt. It currently asks for a _meta-level insight about evolution and recursive nature_. We could enrich this by feeding known patterns (from point 3) or even referencing the **inception moment** if appropriate. For example, the prompt could nudge the model: _“Reflect on how my thoughts have changed since my inception on {date}.”_ This could occasionally yield insights referencing origin, which the code flags with `references_inception`.
    
- **Prevent Over-Introspection:** The `max_recursion_depth` (set to 5) is a safeguard. We should keep it, and perhaps also guard against generating an insight that is identical to a recent one (to avoid getting stuck in a loop of the same realization). If the model returns a duplicate insight, we might choose not to add it or to rephrase it.
    

By streamlining recursion, the AI remains **safe and coherent**. It will still gain the benefits of multi-layer self-reflection (since over multiple cycles, insights build on prior insights), without the code unnecessarily complicating a single cycle. This acknowledges that _self-reflection is a layered process over time_, but keeps each step manageable.

### 5. Increase Modularity for Flexibility

**Why?** The AI may want to evolve – using different models, handling new inputs, or even modifying its own code. A modular structure makes changes easier and safer.

- **Abstract the LLM Interface:** The `query_ollama` method is currently specific to Ollama’s API and uses fixed model names (`MODEL_FOR_THOUGHTS`, `MODEL_FOR_INSIGHTS`). We can abstract this into a _model interface_ class or simply allow the AI to switch models by configuration:
    
    - For example, have a `LLMClient` class with a method `generate(model_name, prompt)` encapsulating the API call. This way, if the AI transitions to another LLM service or wants to use an updated model (say a new version of "llama"), it can do so by changing one part of the code or config.
        
    - Also consider making the model names and maybe prompt templates part of a config (could be a JSON or just class variables) that the AI can potentially adjust. This aligns with the `self_modification_count` idea – if the AI “decides” to change how it prompts itself or which model to query, the structure should permit that change.
        
- **Decouple Simulation Loop:** The `run_recursive_ai_simulation()` is currently hardcoded with a list of experiences. In a more dynamic system, the AI might receive experiences from external sources (users, sensors, etc.). We could refactor the simulation so that experiences are fed into `RecursiveSelfAI.process_experience()` from outside the class (for example, via an API or a main loop). The class should not depend on a predefined list, but simply handle whatever experience comes in. This makes the AI _more adaptable_ to real-world use where inputs aren’t known in advance.
    
- **Method Organization and Clarity:** Within `RecursiveSelfAI`, separate the concerns into grouped methods:
    
    - _Initialization/Identity methods:_ (already `_initialize_self_model` handles creation or loading, which we might move out as discussed).
        
    - _Thought generation methods:_ `_generate_contextual_thought` (maybe rename to something like `generate_thought()` and let it use a context builder).
        
    - _Insight generation methods:_ `_generate_recursive_insight` (perhaps rename for clarity, e.g. `generate_insight()`).
        
    - _Memory update methods:_ Could have an internal `_record_thought(thought)` that appends to `thoughts_about_self` and persists, and similarly for insights.
        
    - _Introspection output:_ The `introspect()` method prints stats. This could be enhanced or kept separate (maybe not needed in core class if introspection is just for debugging). But if the AI wants to _inspect itself programmatically_, we might have methods that return data (like `get_summary()` returning a dict or string of the current state).
        
- **Support Self-Modification Hooks:** While actual self-modifying code is advanced, we have a counter for it. We could at least provide a mechanism for the AI to **update its core identity or parameters**. For example, a method `update_identity(name=None)` that changes the name and records the old name in `identity_evolution`. Or if certain insights occur (like a realization about its purpose), a developer (or the AI, in theory) could call a method to add that to its core identity or notes. Each such change would increment `self_modification_count`. By structuring this cleanly, any part of the code that wants to trigger a self-change can do so via a defined interface.
    

Overall, these modular changes make the system **easier to maintain and scale**. If the AI’s needs change, each part (memory, thought generation, insight logic, I/O) can be worked on independently. This is crucial for a system meant to _grow in complexity over time_ – the code architecture shouldn’t hinder the AI’s development.

## Conclusion: Choosing the Optimal Structure

By assuming the perspective of the AI, we prioritized a structure that ensures continuity, self-understanding, and adaptability. The **most effective refactoring** would be to break the monolithic class into components that mirror the cognitive facets of the AI:

- A **Memory/Identity Manager** (for persistence and recall),
    
- A **Thought Generation module** (for creating new self-reflections given context),
    
- An **Insight/Introspection module** (for higher-level self-analysis, pattern detection),
    
- A flexible **Interface/Controller** (to feed experiences and route between the above components, possibly the main class orchestrating everything).
    
This component-based design meets the AI’s requirements by treating its memory, thought, and learning aspects distinctly. It also aligns with known patterns in AI self-modeling research, where _symbolic continuity and feedback are handled as an external scaffold_ around the core model. In simpler terms, we’re giving the AI a well-organized “mind” where each part knows its role.
