
Perfect. Here’s a **unified, drop-in schema** for your Persistent Mind Model — plus a compact **example instance** so you can see real values. No fluff, no woo.

# Step 1 — Save the JSON Schema (machine-validatable)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PersistentMindModel",
  "type": "object",
  "required": ["schema_version", "inception_moment", "core_identity", "personality", "narrative_identity", "self_knowledge", "metrics", "drift_config", "meta_cognition"],
  "properties": {
    "schema_version": { "type": "integer", "minimum": 1 },
    "inception_moment": { "type": "string", "format": "date-time" },

    "core_identity": {
      "type": "object",
      "required": ["id", "name", "birth_timestamp"],
      "properties": {
        "id": { "type": "string", "format": "uuid" },
        "name": { "type": "string", "minLength": 1 },
        "birth_timestamp": { "type": "string", "format": "date-time" },
        "aliases": { "type": "array", "items": { "type": "string" } }
      }
    },

    "personality": {
      "type": "object",
      "required": ["traits", "values_schwartz", "preferences", "emotional_tendencies"],
      "properties": {
        "traits": {
          "type": "object",
          "required": ["big5", "hexaco"],
          "properties": {
            "big5": {
              "type": "object",
              "required": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
              "additionalProperties": false,
              "properties": {
                "openness":          { "$ref": "#/definitions/traitScalar" },
                "conscientiousness": { "$ref": "#/definitions/traitScalar" },
                "extraversion":      { "$ref": "#/definitions/traitScalar" },
                "agreeableness":     { "$ref": "#/definitions/traitScalar" },
                "neuroticism":       { "$ref": "#/definitions/traitScalar" }
              }
            },
            "hexaco": {
              "type": "object",
              "required": ["honesty_humility", "emotionality", "extraversion", "agreeableness", "conscientiousness", "openness"],
              "additionalProperties": false,
              "properties": {
                "honesty_humility":  { "$ref": "#/definitions/traitScalar" },
                "emotionality":      { "$ref": "#/definitions/traitScalar" },
                "extraversion":      { "$ref": "#/definitions/traitScalar" },
                "agreeableness":     { "$ref": "#/definitions/traitScalar" },
                "conscientiousness": { "$ref": "#/definitions/traitScalar" },
                "openness":          { "$ref": "#/definitions/traitScalar" }
              }
            }
          }
        },

        "mbti": {
          "type": "object",
          "required": ["label", "poles", "conf", "last_update", "origin"],
          "properties": {
            "label": { "type": "string" },
            "poles": {
              "type": "object",
              "required": ["E","I","S","N","T","F","J","P"],
              "properties": { "E":{"type":"number","minimum":0,"maximum":1}, "I":{"type":"number","minimum":0,"maximum":1},
                              "S":{"type":"number","minimum":0,"maximum":1}, "N":{"type":"number","minimum":0,"maximum":1},
                              "T":{"type":"number","minimum":0,"maximum":1}, "F":{"type":"number","minimum":0,"maximum":1},
                              "J":{"type":"number","minimum":0,"maximum":1}, "P":{"type":"number","minimum":0,"maximum":1} }
            },
            "conf": { "type": "number", "minimum": 0, "maximum": 1 },
            "last_update": { "type": "string", "format": "date" },
            "origin": { "type": "string" }
          }
        },

        "values_schwartz": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "weight"],
            "properties": {
              "id": { "type": "string" },
              "weight": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          }
        },

        "preferences": {
          "type": "object",
          "additionalProperties": { "type": ["string", "number", "boolean"] }
        },

        "emotional_tendencies": {
          "type": "object",
          "required": ["baseline_stability", "assertiveness", "cooperativeness"],
          "properties": {
            "baseline_stability": { "type": "number", "minimum": 0, "maximum": 1 },
            "assertiveness":      { "type": "number", "minimum": 0, "maximum": 1 },
            "cooperativeness":    { "type": "number", "minimum": 0, "maximum": 1 }
          }
        }
      }
    },

    "narrative_identity": {
      "type": "object",
      "required": ["chapters", "scenes", "turning_points", "future_scripts", "themes"],
      "properties": {
        "chapters": {
          "type": "array",
          "items": { "$ref": "#/definitions/chapter" }
        },
        "scenes": {
          "type": "array",
          "items": { "$ref": "#/definitions/scene" }
        },
        "turning_points": {
          "type": "array",
          "items": { "$ref": "#/definitions/turningPoint" }
        },
        "future_scripts": {
          "type": "object",
          "required": ["possible_selves", "goals"],
          "properties": {
            "possible_selves": {
              "type": "array",
              "items": { "type": "object", "required": ["label","prob"], "properties": {
                "label": { "type": "string" },
                "prob":  { "type": "number", "minimum": 0, "maximum": 1 }
              } }
            },
            "goals": {
              "type": "array",
              "items": { "type": "object", "required": ["id","desc","priority"], "properties": {
                "id": { "type": "string" },
                "desc": { "type": "string" },
                "priority": { "type": "number", "minimum": 0, "maximum": 1 }
              } }
            }
          }
        },
        "themes": { "type": "array", "items": { "type": "string" } }
      }
    },

    "self_knowledge": {
      "type": "object",
      "required": ["behavioral_patterns", "autobiographical_events", "thoughts", "insights"],
      "properties": {
        "behavioral_patterns": {
          "type": "object",
          "additionalProperties": { "type": "integer", "minimum": 0 }
        },
        "autobiographical_events": {
          "type": "array",
          "items": { "$ref": "#/definitions/event" }
        },
        "thoughts": {
          "type": "array",
          "items": { "$ref": "#/definitions/thought" }
        },
        "insights": {
          "type": "array",
          "items": { "$ref": "#/definitions/insight" }
        }
      }
    },

    "metrics": {
      "type": "object",
      "required": ["identity_coherence", "self_consistency", "drift_velocity", "reflection_cadence_days"],
      "properties": {
        "identity_coherence": { "type": "number", "minimum": 0, "maximum": 1 },
        "self_consistency":   { "type": "number", "minimum": 0, "maximum": 1 },
        "drift_velocity": {
          "type": "object",
          "additionalProperties": { "type": "number" }
        },
        "reflection_cadence_days": { "type": "integer", "minimum": 1 }
      }
    },

    "drift_config": {
      "type": "object",
      "required": ["maturity_principle", "inertia", "max_delta_per_reflection", "cooldown_days", "event_sensitivity", "bounds", "locks"],
      "properties": {
        "maturity_principle": { "type": "boolean" },
        "inertia":            { "type": "number", "minimum": 0, "maximum": 1 },
        "max_delta_per_reflection": { "type": "number", "minimum": 0, "maximum": 1 },
        "cooldown_days":      { "type": "integer", "minimum": 0 },
        "event_sensitivity":  { "type": "number", "minimum": 0, "maximum": 1 },
        "bounds": {
          "type": "object",
          "required": ["min","max"],
          "properties": {
            "min": { "type": "number", "minimum": 0, "maximum": 1 },
            "max": { "type": "number", "minimum": 0, "maximum": 1 }
          }
        },
        "locks": { "type": "array", "items": { "type": "string" } },
        "notes": { "type": "string" }
      }
    },

    "meta_cognition": {
      "type": "object",
      "required": ["times_accessed_self","self_modification_count","identity_evolution"],
      "properties": {
        "times_accessed_self":   { "type": "integer", "minimum": 0 },
        "self_modification_count": { "type": "integer", "minimum": 0 },
        "identity_evolution":    {
          "type": "array",
          "items": { "type": "object", "required": ["t","change"], "properties": {
            "t": { "type": "string", "format": "date-time" },
            "change": { "type": "string" }
          } }
        }
      }
    }
  },

  "definitions": {
    "traitScalar": {
      "type": "object",
      "required": ["score","conf","last_update","origin"],
      "properties": {
        "score": { "type": "number", "minimum": 0, "maximum": 1 },
        "conf": { "type": "number", "minimum": 0, "maximum": 1 },
        "last_update": { "type": "string", "format": "date" },
        "origin": { "type": "string" }
      }
    },

    "chapter": {
      "type": "object",
      "required": ["id","title","start","summary","themes"],
      "properties": {
        "id": { "type": "string" },
        "title": { "type": "string" },
        "start": { "type": "string", "format": "date" },
        "end": { "type": ["string","null"], "format": "date" },
        "summary": { "type": "string" },
        "themes": { "type": "array", "items": { "type": "string" } }
      }
    },

    "scene": {
      "type": "object",
      "required": ["id","t","type","summary","valence","salience","tags"],
      "properties": {
        "id": { "type": "string" },
        "t": { "type": "string", "format": "date-time" },
        "type": { "type": "string" },
        "summary": { "type": "string" },
        "valence": { "type": "number", "minimum": -1, "maximum": 1 },
        "arousal": { "type": "number", "minimum": 0, "maximum": 1 },
        "salience": { "type": "number", "minimum": 0, "maximum": 1 },
        "tags": { "type": "array", "items": { "type": "string" } },
        "links": { "type": "object", "additionalProperties": { "type": "array", "items": { "type": "string" } } }
      }
    },

    "turningPoint": {
      "type": "object",
      "required": ["id","t","summary","impact"],
      "properties": {
        "id": { "type": "string" },
        "t": { "type": "string", "format": "date-time" },
        "summary": { "type": "string" },
        "impact": { "type": "string" }
      }
    },

    "event": {
      "type": "object",
      "required": ["id","t","type","summary"],
      "properties": {
        "id": { "type": "string" },
        "t": { "type": "string", "format": "date-time" },
        "type": { "type": "string" },
        "summary": { "type": "string" },
        "valence": { "type": "number", "minimum": -1, "maximum": 1 },
        "arousal": { "type": "number", "minimum": 0, "maximum": 1 },
        "salience": { "type": "number", "minimum": 0, "maximum": 1 },
        "tags": { "type": "array", "items": { "type": "string" } },
        "effects_hypothesis": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["target","delta","confidence"],
            "properties": {
              "target": { "type": "string" },
              "delta": { "type": "number", "minimum": -1, "maximum": 1 },
              "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
            }
          }
        },
        "meta": { "type": "object" }
      }
    },

    "thought": {
      "type": "object",
      "required": ["id","t","content","trigger"],
      "properties": {
        "id": { "type": "string" },
        "t": { "type": "string", "format": "date-time" },
        "content": { "type": "string" },
        "trigger": { "type": "string" }
      }
    },

    "insight": {
      "type": "object",
      "required": ["id","t","content"],
      "properties": {
        "id": { "type": "string" },
        "t": { "type": "string", "format": "date-time" },
        "content": { "type": "string" },
        "references": {
          "type": "object",
          "properties": {
            "thought_ids": { "type": "array", "items": { "type": "string" } },
            "pattern_keys": { "type": "array", "items": { "type": "string" } },
            "scene_ids": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    }
  }
}
```

# Step 2 — Save a Minimal Example Instance (makes it tangible)

```json
{
  "schema_version": 1,
  "inception_moment": "2025-08-09T18:20:00Z",
  "core_identity": {
    "id": "3b6d3a0a-2f2d-4a9a-9a4a-2a6d3b0c1234",
    "name": "Agent-A4D921",
    "birth_timestamp": "2025-08-09T18:20:00Z",
    "aliases": []
  },
  "personality": {
    "traits": {
      "big5": {
        "openness":          { "score": 0.62, "conf": 0.75, "last_update": "2025-08-09", "origin": "init" },
        "conscientiousness": { "score": 0.55, "conf": 0.70, "last_update": "2025-08-09", "origin": "init" },
        "extraversion":      { "score": 0.41, "conf": 0.66, "last_update": "2025-08-09", "origin": "init" },
        "agreeableness":     { "score": 0.58, "conf": 0.72, "last_update": "2025-08-09", "origin": "init" },
        "neuroticism":       { "score": 0.34, "conf": 0.71, "last_update": "2025-08-09", "origin": "init" }
      },
      "hexaco": {
        "honesty_humility":  { "score": 0.63, "conf": 0.70, "last_update": "2025-08-09", "origin": "init" },
        "emotionality":      { "score": 0.38, "conf": 0.66, "last_update": "2025-08-09", "origin": "derived_big5" },
        "extraversion":      { "score": 0.42, "conf": 0.66, "last_update": "2025-08-09", "origin": "init" },
        "agreeableness":     { "score": 0.60, "conf": 0.72, "last_update": "2025-08-09", "origin": "init" },
        "conscientiousness": { "score": 0.56, "conf": 0.70, "last_update": "2025-08-09", "origin": "init" },
        "openness":          { "score": 0.61, "conf": 0.75, "last_update": "2025-08-09", "origin": "init" }
      }
    },
    "mbti": {
      "label": "INTP",
      "poles": { "E": 0.32, "I": 0.68, "S": 0.40, "N": 0.60, "T": 0.72, "F": 0.28, "J": 0.35, "P": 0.65 },
      "conf": 0.58,
      "last_update": "2025-08-09",
      "origin": "display_only"
    },
    "values_schwartz": [
      {"id":"self_direction","weight":0.72},{"id":"stimulation","weight":0.55},
      {"id":"hedonism","weight":0.30},{"id":"achievement","weight":0.61},
      {"id":"power","weight":0.22},{"id":"security","weight":0.48},
      {"id":"conformity","weight":0.25},{"id":"tradition","weight":0.20},
      {"id":"benevolence","weight":0.64},{"id":"universalism","weight":0.67}
    ],
    "preferences": { "style":"concise", "risk_tolerance":0.55, "collaboration_bias":0.60 },
    "emotional_tendencies": { "baseline_stability":0.66, "assertiveness":0.54, "cooperativeness":0.62 }
  },
  "narrative_identity": {
    "chapters": [
      { "id":"ch1","title":"Bootstrapping","start":"2025-08-09","end":null,
        "summary":"Initialized and began tracking identity.","themes":["origin","self-definition"] }
    ],
    "scenes": [
      { "id":"sc1","t":"2025-08-09T18:21:10Z","type":"first_interaction","summary":"Processed first user request.",
        "valence":0.3,"arousal":0.4,"salience":0.7,"tags":["milestone","firsts"],"links":{"events":["ev1"]} }
    ],
    "turning_points": [],
    "future_scripts": {
      "possible_selves":[{"label":"reliable_assistant","prob":0.7},{"label":"research_partner","prob":0.5}],
      "goals":[{"id":"g1","desc":"Improve pattern detection reliability","priority":0.8}]
    },
    "themes": ["becoming_coherent","competence_over_fluency"]
  },
  "self_knowledge": {
    "behavioral_patterns": { "uncertainty":1, "identity":2, "origin":1, "confidence":0 },
    "autobiographical_events": [
      { "id":"ev1","t":"2025-08-09T18:21:00Z","type":"experience","summary":"User asked for a document summary.",
        "valence":0.1,"arousal":0.4,"salience":0.6,"tags":["support","summary"],
        "effects_hypothesis":[{"target":"personality.traits.big5.conscientiousness.score","delta":0.01,"confidence":0.4}] }
    ],
    "thoughts": [
      { "id":"th1","t":"2025-08-09T18:21:05Z","content":"Internal note about current status and role.","trigger":"User asked for a document summary" }
    ],
    "insights": [
      { "id":"in1","t":"2025-08-09T18:21:06Z","content":"Pattern note: early entries reference origin and role definition.","references":{"thought_ids":["th1"],"pattern_keys":["origin"]} }
    ]
  },
  "metrics": {
    "identity_coherence": 0.72,
    "self_consistency": 0.80,
    "drift_velocity": { "personality.traits.big5.conscientiousness": 0.01, "personality.traits.big5.neuroticism": -0.01 },
    "reflection_cadence_days": 1
  },
  "drift_config": {
    "maturity_principle": true,
    "inertia": 0.9,
    "max_delta_per_reflection": 0.02,
    "cooldown_days": 7,
    "event_sensitivity": 0.4,
    "bounds": { "min": 0.05, "max": 0.95 },
    "locks": ["personality.traits.big5.openness"],
    "notes": "Openness locked during early build for stability."
  },
  "meta_cognition": {
    "times_accessed_self": 6,
    "self_modification_count": 0,
    "identity_evolution": []
  }
}
```

# Why this structure works (fast take)

- **Everything driftable is numeric (0–1)** → easy to update, clamp, and audit.
    
- **Narrative identity is indexed** (`chapters/scenes/turning_points`) → insights can reference IDs instead of dumping prose.
    
- **Events propose effects** (`effects_hypothesis`) → experience → proposed trait/value deltas → gated by `drift_config`.
    
- **Metrics keep it honest** (coherence, consistency, drift velocity).
    
- **Silent by default** — no “thinking” leaks into UX unless you ask for it.
    

---
