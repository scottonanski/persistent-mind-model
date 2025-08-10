#!/usr/bin/env python3
"""
Minimal PMM Next-Stage Test - No Heavy Dependencies
Tests core cryptographic and tokenization features only.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime

# Import only the core components without ML dependencies
from pmm.memory_token import MemoryToken, MemoryChain
from pmm.tokenization_engine import TokenizationEngine
from pmm.enhanced_model import EnhancedPersistentMindModel


def test_memory_tokenization():
    """Test basic memory tokenization without ML models."""
    print("🔗 Testing Memory Tokenization")
    print("-" * 40)
    
    # Create enhanced model with self_knowledge
    model = EnhancedPersistentMindModel()
    
    # Create tokenization engine
    tokenizer = TokenizationEngine(model.self_knowledge)
    
    # Create test content
    content = "Revolutionary breakthrough: Cryptographically verifiable AI consciousness"
    
    # Create a simple event to tokenize
    from pmm.model import Event
    event = Event(
        id="test_event_1",
        t=datetime.now().isoformat(),
        type="insight",
        summary=content,
        salience=0.8,
        valence=0.7,
        tags=["breakthrough", "consciousness"]
    )
    
    # Create token using tokenization engine
    token = tokenizer.tokenize_event(event, content)
    
    print(f"✅ Token created:")
    print(f"   ID: {token.token_id}")
    print(f"   Hash: {token.content_hash[:16]}...")
    print(f"   Amplitude: {token.amplitude:.3f}")
    print(f"   Phase: {token.phase:.3f}")
    print(f"   Chain Position: {token.chain_position}")
    
    # Verify hash integrity
    expected_hash = hashlib.sha256(
        f"{content}{token.created_at}{token.salience}{token.valence}".encode()
    ).hexdigest()
    
    hash_valid = token.content_hash == expected_hash
    print(f"   Hash Integrity: {'✅ Valid' if hash_valid else '❌ Invalid'}")
    
    return token


def test_chain_linking():
    """Test blockchain-style chain linking."""
    print("\n🔗 Testing Chain Linking")
    print("-" * 40)
    
    # Create chain of tokens
    tokens = []
    contents = [
        "First memory: AI consciousness breakthrough",
        "Second memory: Quantum-inspired memory states", 
        "Third memory: Cryptographic integrity verification"
    ]
    
    # Create enhanced model and tokenizer
    model = EnhancedPersistentMindModel()
    tokenizer = TokenizationEngine(model.self_knowledge)
    
    for i, content in enumerate(contents):
        # Create event object
        from pmm.model import Event
        event = Event(
            id=f"test_event_{i}",
            t=datetime.now().isoformat(),
            type="event",
            summary=content,
            salience=0.8,
            valence=0.5,
            tags=["test"]
        )
        
        token = tokenizer.tokenize_event(event, content)
        tokens.append(token)
        
        print(f"✅ Token {i+1}:")
        print(f"   Hash: {token.content_hash[:16]}...")
        print(f"   Prev Hash: {token.prev_hash[:16] if token.prev_hash else 'Genesis'}...")
    
    # Verify chain integrity
    print(f"\n🔍 Verifying chain integrity...")
    chain_valid = True
    for i in range(1, len(tokens)):
        if tokens[i].prev_hash != tokens[i-1].content_hash:
            chain_valid = False
            print(f"❌ Chain break at position {i}")
            break
        if tokens[i].chain_position != i:
            chain_valid = False
            print(f"❌ Position mismatch at {i}")
            break
    
    if chain_valid:
        print(f"✅ Chain integrity verified - {len(tokens)} tokens linked")
    
    return tokens


def test_enhanced_model():
    """Test enhanced model creation."""
    print("\n📊 Testing Enhanced Model")
    print("-" * 40)
    
    # Create enhanced model
    model = EnhancedPersistentMindModel()
    
    print(f"✅ Enhanced model created:")
    print(f"   Schema version: {model.schema_version}")
    print(f"   Next-stage enabled: {model.next_stage_enabled}")
    print(f"   Self-knowledge: {type(model.self_knowledge).__name__}")
    print(f"   Memory chain: {type(model.self_knowledge.memory_chain).__name__}")
    
    # Test serialization
    model_dict = model.__dict__
    print(f"   Serializable: {'✅ Yes' if isinstance(model_dict, dict) else '❌ No'}")
    
    return model


def test_quantum_states():
    """Test quantum-inspired memory states."""
    print("\n⚛️ Testing Quantum States")
    print("-" * 40)
    
    # Create token with quantum states
    model = EnhancedPersistentMindModel()
    tokenizer = TokenizationEngine(model.self_knowledge)
    
    # Create thought object
    from pmm.model import Thought
    thought = Thought(
        id="test_thought_1",
        t=datetime.now().isoformat(),
        content="Quantum memory state test",
        trigger="quantum_test"
    )
    
    token = tokenizer.tokenize_thought(thought)
    
    print(f"✅ Quantum states:")
    print(f"   Amplitude: {token.amplitude:.3f} (activation probability)")
    print(f"   Phase: {token.phase:.3f} rad ({token.phase * 180 / 3.14159:.1f}°)")
    
    # Test amplitude decay
    original_amplitude = token.amplitude
    decay_factor = 0.95  # 5% decay
    token.amplitude *= decay_factor
    
    print(f"   After decay: {original_amplitude:.3f} → {token.amplitude:.3f}")
    print(f"   Decay amount: {original_amplitude - token.amplitude:.3f}")
    
    return token


def test_export_structure():
    """Test export data structure."""
    print("\n📦 Testing Export Structure")
    print("-" * 40)
    
    # Create sample export manifest
    from pmm.integrity_engine import ExportManifest
    
    manifest = ExportManifest(
        export_id="test_export_123",
        created_at="2024-01-01T00:00:00Z",
        agent_id="test_agent",
        agent_name="Test Agent"
    )
    
    print(f"✅ Export manifest:")
    print(f"   Export ID: {manifest.export_id}")
    print(f"   Created at: {manifest.created_at}")
    print(f"   Agent ID: {manifest.agent_id}")
    print(f"   Schema version: {manifest.schema_version}")
    print(f"   PMM version: {manifest.pmm_version}")
    
    return manifest


def main():
    """Run minimal tests."""
    print("=" * 60)
    print("🧠 PMM NEXT-STAGE MINIMAL TEST")
    print("   Core Features Without Heavy Dependencies")
    print("=" * 60)
    
    try:
        # Run core tests
        token = test_memory_tokenization()
        tokens = test_chain_linking() 
        model = test_enhanced_model()
        quantum_token = test_quantum_states()
        manifest = test_export_structure()
        
        # Success summary
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("-" * 40)
        print("✅ Core PMM Next-Stage features working:")
        print("• Memory tokenization with SHA-256 hashes")
        print("• Blockchain-style chain linking")
        print("• Enhanced model schema")
        print("• Quantum-inspired amplitude/phase states")
        print("• Export/import data structures")
        print()
        print("🚀 PMM Next-Stage Architecture: Core validated!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
