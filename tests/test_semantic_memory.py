#!/usr/bin/env python3
"""
Test script to validate semantic memory search implementation.

Tests:
1. Embedding generation and storage
2. Semantic similarity search
3. Hybrid memory retrieval (semantic + chronological)
4. Context relevance improvement
"""

import tempfile
import os
import numpy as np
from pmm.langchain_memory import PersistentMindMemory
from pmm.semantic_analysis import get_semantic_analyzer


def test_embedding_generation():
    """Test that embeddings are generated and stored correctly."""
    print("🧪 Testing embedding generation and storage...")

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_path = os.path.join(tmpdir, "test_agent.json")

        # Create memory with embeddings enabled
        memory = PersistentMindMemory(agent_path=agent_path, enable_embeddings=True)

        # Save some conversation context
        test_inputs = [
            "What is machine learning?",
            "How do neural networks work?",
            "Tell me about Python programming",
            "What's the weather like today?",
        ]

        test_outputs = [
            "Machine learning is a subset of AI that enables computers to learn from data.",
            "Neural networks are computational models inspired by biological neural networks.",
            "Python is a high-level programming language known for its simplicity.",
            "I don't have access to current weather data, but I can help with other questions.",
        ]

        # Store conversations with embeddings
        for i, (inp, out) in enumerate(zip(test_inputs, test_outputs)):
            memory.save_context(inputs={"input": inp}, outputs={"response": out})

        # Check that events have embeddings (user inputs embed synchronously)
        events_with_embeddings = memory.pmm.sqlite_store.get_events_with_embeddings()
        assert events_with_embeddings, "No events with embeddings found"

        print(f"✅ Generated embeddings for {len(events_with_embeddings)} events")

        # Verify embedding format
        for event in events_with_embeddings[:2]:
            embedding = event.get("embedding")
            if embedding:
                # Convert bytes back to numpy array
                embedding_array = np.frombuffer(embedding, dtype=np.float32)
                if len(embedding_array) > 0:
                    print(f"✅ Embedding shape: {embedding_array.shape}")
                else:
                    raise AssertionError("Empty embedding array")


def test_semantic_search():
    """Test semantic similarity search functionality."""
    print("🧪 Testing semantic similarity search...")

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_path = os.path.join(tmpdir, "test_agent.json")

        memory = PersistentMindMemory(agent_path=agent_path, enable_embeddings=True)

        # Store diverse conversations
        conversations = [
            (
                "What is artificial intelligence?",
                "AI is the simulation of human intelligence in machines.",
            ),
            (
                "How do I cook pasta?",
                "Boil water, add pasta, cook for 8-12 minutes until al dente.",
            ),
            (
                "Explain machine learning algorithms",
                "ML algorithms learn patterns from data to make predictions.",
            ),
            (
                "What's your favorite color?",
                "I don't have personal preferences, but I can discuss color theory.",
            ),
            (
                "Tell me about deep learning",
                "Deep learning uses neural networks with multiple layers.",
            ),
        ]

        for inp, out in conversations:
            memory.save_context(inputs={"input": inp}, outputs={"response": out})

        # Test semantic search with AI-related query
        ai_query = "Tell me about artificial neural networks"
        semantic_analyzer = get_semantic_analyzer()
        query_embedding_list = semantic_analyzer._get_embedding(ai_query)
        query_embedding = np.array(query_embedding_list, dtype=np.float32).tobytes()

        # Search for similar events
        similar_events = memory.pmm.sqlite_store.semantic_search(
            query_embedding, limit=3
        )

        assert similar_events, "No similar events found"

        print(f"✅ Found {len(similar_events)} semantically similar events")

        # Check that AI-related conversations are ranked higher
        ai_related_found = False
        for event in similar_events:
            content = event.get("content", "").lower()
            if any(
                term in content
                for term in [
                    "artificial",
                    "intelligence",
                    "machine",
                    "learning",
                    "neural",
                ]
            ):
                ai_related_found = True
                print(f"✅ AI-related content found: {content[:50]}...")
                break

        assert ai_related_found, "Expected AI-related content not found in top results"


def test_hybrid_memory_retrieval():
    """Test hybrid memory retrieval combining semantic and chronological context."""
    print("🧪 Testing hybrid memory retrieval...")

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_path = os.path.join(tmpdir, "test_agent.json")

        memory = PersistentMindMemory(agent_path=agent_path, enable_embeddings=True)

        # Store a series of conversations
        conversations = [
            ("What is Python?", "Python is a programming language."),
            ("How's the weather?", "I don't have weather data."),
            (
                "Tell me about programming",
                "Programming involves writing code to solve problems.",
            ),
            ("What's for lunch?", "I can't see what you're having for lunch."),
            ("Explain Python syntax", "Python uses indentation and simple syntax."),
        ]

        for inp, out in conversations:
            memory.save_context(inputs={"input": inp}, outputs={"response": out})

        # Test memory loading with programming-related query
        programming_query = "How do I write Python code?"

        # Load memory variables (this should use semantic search)
        memory_vars = memory.load_memory_variables({"input": programming_query})

        history = memory_vars.get("history", "")

        assert history, "No history loaded"

        print(f"✅ Loaded memory context ({len(history)} characters)")

        # Check that programming-related content is included
        programming_terms = ["python", "programming", "code", "syntax"]
        programming_mentions = sum(
            1 for term in programming_terms if term.lower() in history.lower()
        )

        assert programming_mentions > 0, "No programming-related content found in memory context"

        print(
            f"✅ Found {programming_mentions} programming-related mentions in context"
        )

        # Check for semantic relevance markers
        if "[Relevant]" in history:
            print("✅ Semantic relevance markers found")
        else:
            print(
                "⚠️  No semantic relevance markers found (may be expected if no embeddings)"
            )

        # Passes if assertions above hold


def test_context_improvement():
    """Test that semantic search improves context relevance."""
    print("🧪 Testing context relevance improvement...")

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_path = os.path.join(tmpdir, "test_agent.json")

        # Test with embeddings disabled (baseline)
        memory_baseline = PersistentMindMemory(
            agent_path=agent_path, enable_embeddings=False
        )

        # Test with embeddings enabled
        memory_semantic = PersistentMindMemory(
            agent_path=agent_path + "_semantic", enable_embeddings=True
        )

        # Store the same conversations in both
        conversations = [
            ("What's the capital of France?", "The capital of France is Paris."),
            ("How do I bake a cake?", "Mix ingredients, bake at 350°F for 30 minutes."),
            (
                "Tell me about Paris",
                "Paris is known for the Eiffel Tower and rich culture.",
            ),
            ("What's 2+2?", "2+2 equals 4."),
            (
                "Describe French cuisine",
                "French cuisine is renowned for its sophistication.",
            ),
        ]

        for inp, out in conversations:
            memory_baseline.save_context(
                inputs={"input": inp}, outputs={"response": out}
            )
            memory_semantic.save_context(
                inputs={"input": inp}, outputs={"response": out}
            )

        # Query about France (should find Paris and French cuisine)
        france_query = "What do you know about France?"

        baseline_vars = memory_baseline.load_memory_variables({"input": france_query})
        semantic_vars = memory_semantic.load_memory_variables({"input": france_query})

        baseline_history = baseline_vars.get("history", "")
        semantic_history = semantic_vars.get("history", "")

        # Count France-related mentions
        france_terms = ["france", "paris", "french"]

        baseline_mentions = sum(
            1 for term in france_terms if term.lower() in baseline_history.lower()
        )
        semantic_mentions = sum(
            1 for term in france_terms if term.lower() in semantic_history.lower()
        )

        print(f"✅ Baseline mentions: {baseline_mentions}")
        print(f"✅ Semantic mentions: {semantic_mentions}")

        # Semantic search should find more relevant content (or at least not worse)
        assert (
            semantic_mentions >= baseline_mentions
        ), "Semantic memory did not improve relevance"


def main():
    """Run all semantic memory tests."""
    print("🚀 Running semantic memory search validation...\n")

    tests = [
        test_embedding_generation,
        test_semantic_search,
        test_hybrid_memory_retrieval,
        test_context_improvement,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            print()

    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("🎉 Semantic memory search implementation validated!")
    else:
        print("⚠️  Some tests failed - implementation needs attention")


if __name__ == "__main__":
    main()
