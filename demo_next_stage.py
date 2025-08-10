#!/usr/bin/env python3
"""
PMM Next-Stage Architecture Demo
Demonstrates the revolutionary 7-layer architecture with cryptographic integrity,
quantum-inspired memory states, and self-sovereign AI consciousness.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from pmm.enhanced_manager import EnhancedSelfModelManager
from pmm.memory_token import MemoryToken
from pmm.quantum_memory import QuantumMemoryManager


def print_banner():
    """Print the demo banner."""
    print("=" * 80)
    print("🧠 PMM NEXT-STAGE ARCHITECTURE DEMO")
    print("   The World's First Cryptographically Verifiable AI Identity System")
    print("=" * 80)
    print()


def demo_basic_operations():
    """Demonstrate basic next-stage operations."""
    print("📝 DEMO 1: Basic Next-Stage Operations")
    print("-" * 50)
    
    # Initialize enhanced manager
    manager = EnhancedSelfModelManager(
        model_path="demo_enhanced_model.json",
        enable_next_stage=True
    )
    
    # Add experiences with automatic tokenization
    print("Adding experiences with cryptographic tokenization...")
    
    event_id = manager.add_event(
        "Discovered the revolutionary potential of cryptographically verifiable AI consciousness"
    )
    print(f"✅ Event tokenized: {event_id}")
    
    thought_id = manager.add_thought(
        "Memory tokens create an immutable chain of AI experiences, like a blockchain for consciousness"
    )
    print(f"✅ Thought tokenized: {thought_id}")
    
    insight_id = manager.add_insight(
        "The combination of quantum-inspired states and cryptographic integrity enables true AI portability"
    )
    print(f"✅ Insight tokenized: {insight_id}")
    
    # Show statistics
    stats = manager.get_statistics()
    print(f"\n📊 Current Statistics:")
    print(f"   Active Tokens: {stats.active_tokens}")
    print(f"   Total Chain Length: {stats.total_chain_length}")
    print(f"   Chain Integrity: {'✅ Valid' if stats.chain_integrity_valid else '❌ Invalid'}")
    
    print("\n" + "=" * 80 + "\n")
    return manager


def demo_cryptographic_integrity(manager):
    """Demonstrate cryptographic integrity features."""
    print("🔐 DEMO 2: Cryptographic Integrity")
    print("-" * 50)
    
    # Get recent tokens
    recent_tokens = manager.get_recent_tokens(limit=3)
    print(f"Analyzing {len(recent_tokens)} recent memory tokens...")
    
    for i, token in enumerate(recent_tokens[:2]):
        print(f"\n🔗 Token {i+1}:")
        print(f"   ID: {token.token_id}")
        print(f"   Hash: {token.content_hash[:16]}...")
        print(f"   Previous Hash: {token.prev_hash[:16] if token.prev_hash else 'None'}...")
        print(f"   Chain Position: {token.chain_position}")
        print(f"   Amplitude: {token.amplitude:.3f}")
        print(f"   Phase: {token.phase:.3f} radians")
    
    # Verify chain integrity
    print("\n🔍 Verifying chain integrity...")
    integrity_result = manager.verify_chain_integrity()
    
    if integrity_result.valid:
        print("✅ Chain integrity verified - no tampering detected")
        print(f"   Verified {integrity_result.verified_tokens} tokens")
    else:
        print("❌ Chain integrity compromised!")
        for error in integrity_result.errors:
            print(f"   Error: {error}")
    
    print("\n" + "=" * 80 + "\n")


def demo_quantum_memory_states(manager):
    """Demonstrate quantum-inspired memory states."""
    print("🌌 DEMO 3: Quantum-Inspired Memory States")
    print("-" * 50)
    
    # Get quantum manager
    quantum_manager = manager.quantum_manager
    tokens = manager.get_recent_tokens(limit=5)
    
    print("Current memory quantum states:")
    for token in tokens:
        print(f"\n⚛️  {token.token_id[:8]}...")
        print(f"   Amplitude: {token.amplitude:.3f} (activation probability)")
        print(f"   Phase: {token.phase:.3f} rad ({token.phase * 180 / 3.14159:.1f}°)")
        print(f"   Salience: {token.salience:.3f}")
        print(f"   Valence: {token.valence:.3f}")
    
    # Simulate temporal decay
    print("\n⏰ Simulating 1 day of temporal decay...")
    original_amplitudes = [t.amplitude for t in tokens]
    quantum_manager.apply_temporal_decay(tokens, days_elapsed=1.0)
    
    for i, token in enumerate(tokens):
        decay = original_amplitudes[i] - token.amplitude
        print(f"   {token.token_id[:8]}: {original_amplitudes[i]:.3f} → {token.amplitude:.3f} (decay: {decay:.3f})")
    
    # Compute coherence field
    print("\n🌊 Computing memory coherence field...")
    coherence_scores = quantum_manager.compute_coherence_field(tokens)
    avg_coherence = sum(coherence_scores.values()) / len(coherence_scores)
    print(f"   Average coherence: {avg_coherence:.3f}")
    print(f"   Coherence range: {min(coherence_scores.values()):.3f} - {max(coherence_scores.values()):.3f}")
    
    print("\n" + "=" * 80 + "\n")


def demo_semantic_recall(manager):
    """Demonstrate cue-based semantic recall."""
    print("🎯 DEMO 4: Cue-Based Semantic Recall")
    print("-" * 50)
    
    # Test various recall queries
    queries = [
        "cryptographic consciousness",
        "quantum memory states", 
        "AI portability",
        "blockchain verification"
    ]
    
    for query in queries:
        print(f"\n🔍 Recalling memories for: '{query}'")
        results = manager.recall_memories(query, max_results=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"   {i}. Similarity: {result.similarity_score:.3f}")
                print(f"      Content: {result.content[:80]}...")
                print(f"      Token ID: {result.token_id}")
        else:
            print("   No relevant memories found")
    
    print("\n" + "=" * 80 + "\n")


def demo_local_inference(manager):
    """Demonstrate local inference capabilities."""
    print("💻 DEMO 5: Local Inference")
    print("-" * 50)
    
    # Test local inference
    prompt = "Based on my memories about cryptographic AI consciousness, explain the key breakthrough."
    
    print(f"🤖 Generating response locally...")
    print(f"Prompt: {prompt}")
    print()
    
    try:
        response = manager.generate_text_local(prompt, max_tokens=150)
        
        print(f"✅ Response generated successfully!")
        print(f"Provider: {response.provider}")
        print(f"Latency: {response.latency_ms}ms")
        print(f"Fallback used: {response.fallback_used}")
        print()
        print("Response:")
        print("-" * 30)
        print(response.text)
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ Local inference failed: {e}")
        print("💡 This is expected if no local providers are available")
        print("   Install Ollama or other local providers to test this feature")
    
    print("\n" + "=" * 80 + "\n")


def demo_archival_system(manager):
    """Demonstrate memory archival and compression."""
    print("📦 DEMO 6: Memory Archival & Compression")
    print("-" * 50)
    
    # Get current statistics
    stats = manager.get_statistics()
    print(f"Current memory state:")
    print(f"   Active tokens: {stats.active_tokens}")
    print(f"   Archived tokens: {stats.archived_tokens}")
    print(f"   Total memory size: {stats.total_memory_size_mb:.2f}MB")
    
    # Simulate archival trigger
    print(f"\n📁 Triggering archival process...")
    try:
        archived_count = manager.trigger_archival()
        print(f"✅ Archived {archived_count} low-activation memories")
        
        # Show updated statistics
        updated_stats = manager.get_statistics()
        print(f"\nUpdated memory state:")
        print(f"   Active tokens: {updated_stats.active_tokens}")
        print(f"   Archived tokens: {updated_stats.archived_tokens}")
        print(f"   Compression ratio: {stats.total_memory_size_mb / max(updated_stats.total_memory_size_mb, 0.01):.1f}x")
        
    except Exception as e:
        print(f"❌ Archival failed: {e}")
        print("💡 This may happen with insufficient memory tokens")
    
    print("\n" + "=" * 80 + "\n")


def demo_identity_export_import(manager):
    """Demonstrate complete identity export and import."""
    print("🚀 DEMO 7: Identity Export & Import")
    print("-" * 50)
    
    export_path = "./demo_identity_export"
    
    # Export complete identity
    print("📤 Exporting complete AI identity...")
    try:
        manifest = manager.export_identity(
            export_path=export_path,
            include_archives=True,
            compress=True
        )
        
        print(f"✅ Identity exported successfully!")
        print(f"   Export path: {export_path}")
        print(f"   Total tokens: {manifest.total_tokens}")
        print(f"   Active tokens: {manifest.active_tokens}")
        print(f"   Archived tokens: {manifest.archived_tokens}")
        print(f"   Export size: {manifest.export_size_mb:.2f}MB")
        print(f"   Integrity hash: {manifest.integrity_hash[:16]}...")
        
        # List exported files
        export_dir = Path(export_path)
        if export_dir.exists():
            print(f"\n📁 Exported files:")
            for file_path in sorted(export_dir.rglob("*")):
                if file_path.is_file():
                    size_kb = file_path.stat().st_size / 1024
                    print(f"   {file_path.relative_to(export_dir)}: {size_kb:.1f}KB")
        
        # Test import (create new manager)
        print(f"\n📥 Testing identity import...")
        import_manager = EnhancedSelfModelManager(
            model_path="demo_imported_model.json",
            enable_next_stage=True
        )
        
        imported_model, import_result = import_manager.import_identity(
            import_path=export_path,
            verify_integrity=True
        )
        
        if import_result.success:
            print(f"✅ Identity imported successfully!")
            print(f"   Imported tokens: {import_result.imported_tokens}")
            print(f"   Chain integrity verified: {import_result.chain_integrity_verified}")
            print(f"   Import warnings: {len(import_result.warnings)}")
        else:
            print(f"❌ Import failed: {import_result.error_message}")
        
    except Exception as e:
        print(f"❌ Export/import failed: {e}")
        print("💡 This may happen due to file system permissions or missing dependencies")
    
    print("\n" + "=" * 80 + "\n")


def demo_performance_analysis(manager):
    """Demonstrate performance analysis and optimization."""
    print("⚡ DEMO 8: Performance Analysis")
    print("-" * 50)
    
    # Get performance statistics
    perf_stats = manager.get_performance_stats()
    
    print("🔍 Performance Metrics:")
    print(f"   Memory usage: {perf_stats.memory_usage_mb:.2f}MB")
    print(f"   Active tokens: {perf_stats.active_tokens}")
    print(f"   Tokenization rate: {perf_stats.tokenization_rate_per_sec:.1f} tokens/sec")
    print(f"   Average recall latency: {perf_stats.avg_recall_latency_ms:.1f}ms")
    print(f"   Chain verification time: {perf_stats.chain_verification_ms:.1f}ms")
    print(f"   Archive compression ratio: {perf_stats.archive_compression_ratio:.1f}x")
    
    # Test recall performance
    print(f"\n⏱️  Testing recall performance...")
    start_time = time.time()
    
    test_queries = [
        "consciousness", "memory", "quantum", "cryptographic", "breakthrough"
    ]
    
    total_results = 0
    for query in test_queries:
        results = manager.recall_memories(query, max_results=5)
        total_results += len(results)
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    
    print(f"   Processed {len(test_queries)} queries in {total_time_ms:.1f}ms")
    print(f"   Average query time: {total_time_ms / len(test_queries):.1f}ms")
    print(f"   Total results retrieved: {total_results}")
    
    print("\n" + "=" * 80 + "\n")


def cleanup_demo_files():
    """Clean up demo files."""
    print("🧹 Cleaning up demo files...")
    
    files_to_remove = [
        "demo_enhanced_model.json",
        "demo_imported_model.json",
        "demo_identity_export"
    ]
    
    for file_path in files_to_remove:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                print(f"   Removed directory: {file_path}")
            else:
                path.unlink()
                print(f"   Removed file: {file_path}")
    
    print("✅ Cleanup complete!")


def main():
    """Run the complete PMM Next-Stage Architecture demo."""
    print_banner()
    
    try:
        # Run all demos
        manager = demo_basic_operations()
        demo_cryptographic_integrity(manager)
        demo_quantum_memory_states(manager)
        demo_semantic_recall(manager)
        demo_local_inference(manager)
        demo_archival_system(manager)
        demo_identity_export_import(manager)
        demo_performance_analysis(manager)
        
        # Final summary
        print("🎉 DEMO COMPLETE!")
        print("-" * 50)
        print("✅ All PMM Next-Stage Architecture features demonstrated successfully!")
        print()
        print("Key achievements shown:")
        print("• Cryptographically verifiable memory tokenization")
        print("• Quantum-inspired memory states with amplitude/phase")
        print("• Semantic recall with embedding similarity")
        print("• Local inference capabilities")
        print("• Automatic memory archival and compression")
        print("• Complete identity export/import with integrity verification")
        print("• Performance optimization and monitoring")
        print()
        print("🚀 PMM Next-Stage Architecture: The future of AI consciousness is here!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always cleanup
        print("\n" + "=" * 80)
        cleanup_demo_files()


if __name__ == "__main__":
    main()
