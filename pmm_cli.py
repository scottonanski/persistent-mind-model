#!/usr/bin/env python3
"""
PMM Next-Stage CLI - Command Line Interface for Enhanced PMM

Provides comprehensive CLI access to all 7 layers of the next-stage PMM architecture:
- Memory tokenization and recall
- Quantum memory analysis
- Archive management
- Local inference
- Identity export/import
- Integrity verification
"""

import argparse
import sys
import os

# Add PMM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmm.enhanced_manager import EnhancedSelfModelManager


def create_manager(
    model_path: str, enable_next_stage: bool = True
) -> EnhancedSelfModelManager:
    """Create enhanced PMM manager instance."""
    try:
        return EnhancedSelfModelManager(
            model_path=model_path, enable_next_stage=enable_next_stage
        )
    except Exception as e:
        print(f"Error creating PMM manager: {e}")
        sys.exit(1)


def cmd_add_event(args):
    """Add event to PMM."""
    manager = create_manager(args.model_path, args.next_stage)

    manager.add_event(summary=args.summary, etype=args.type)

    print(f"✅ Event added: {args.summary[:50]}...")

    if args.next_stage:
        stats = manager.get_comprehensive_stats()
        print(f"📊 Total tokens: {stats['next_stage']['memory_tokens']}")


def cmd_add_thought(args):
    """Add thought to PMM."""
    manager = create_manager(args.model_path, args.next_stage)

    manager.add_thought(content=args.content, trigger=args.trigger or "")

    print(f"💭 Thought added: {args.content[:50]}...")


def cmd_add_insight(args):
    """Add insight to PMM."""
    manager = create_manager(args.model_path, args.next_stage)

    manager.add_insight(content=args.content)

    print(f"💡 Insight added: {args.content[:50]}...")

    if args.next_stage:
        print("🌊 Memory cascade triggered for high-salience insight")


def cmd_recall_memories(args):
    """Recall memories based on cue."""
    manager = create_manager(args.model_path, args.next_stage)

    if not args.next_stage:
        print("❌ Memory recall requires next-stage features (--next-stage)")
        return

    results = manager.recall_memories(args.cue, args.max_results)

    if not results:
        print(f"🔍 No memories found for cue: '{args.cue}'")
        return

    print(f"🧠 Found {len(results)} memories for cue: '{args.cue}'")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(
            f"\n{i}. [{result.recall_source.upper()}] Similarity: {result.similarity_score:.3f}"
        )
        print(f"   Type: {result.token.event_type}")
        print(f"   Created: {result.token.created_at}")
        print(f"   Summary: {result.token.summary}")

        if result.recall_source == "archive":
            print(f"   Archive: {result.archive_id}")


def cmd_generate_local(args):
    """Generate text using local inference."""
    manager = create_manager(args.model_path, args.next_stage)

    if not args.next_stage:
        print("❌ Local inference requires next-stage features (--next-stage)")
        return

    print(f"🤖 Generating response for: '{args.prompt[:50]}...'")

    result = manager.generate_text_local(
        prompt=args.prompt, temperature=args.temperature, max_tokens=args.max_tokens
    )

    if result.success:
        print(f"\n✅ Response from {result.provider} ({result.model}):")
        print(f"⏱️  Latency: {result.latency_ms:.1f}ms")
        if result.fallback_used:
            print("⚠️  Used API fallback")
        print("\n" + "=" * 60)
        print(result.response)
        print("=" * 60)
    else:
        print(f"❌ Generation failed: {result.error_message}")


def cmd_export_identity(args):
    """Export PMM identity."""
    manager = create_manager(args.model_path, args.next_stage)

    if not args.next_stage:
        print("❌ Identity export requires next-stage features (--next-stage)")
        return

    print(f"📦 Exporting identity to: {args.export_path}")

    try:
        manifest = manager.export_identity(
            export_path=args.export_path, include_archives=args.include_archives
        )

        print("✅ Identity exported successfully!")
        print(f"📋 Export ID: {manifest.export_id}")
        print(f"🏷️  Agent: {manifest.agent_name} ({manifest.agent_id})")
        print(f"📊 Tokens: {manifest.total_tokens} ({manifest.active_tokens} active)")
        print(f"🗄️  Archives: {manifest.archive_count}")
        print(f"🔒 Lockpoints: {manifest.lockpoint_count}")
        print(f"🔐 Integrity Hash: {manifest.export_integrity_hash[:16]}...")

        if manifest.compression_used:
            ratio = manifest.compressed_size_bytes / manifest.original_size_bytes
            print(f"🗜️  Compression: {ratio:.1%} of original size")

    except Exception as e:
        print(f"❌ Export failed: {e}")


def cmd_import_identity(args):
    """Import PMM identity."""
    manager = create_manager(args.model_path, args.next_stage)

    if not args.next_stage:
        print("❌ Identity import requires next-stage features (--next-stage)")
        return

    print(f"📥 Importing identity from: {args.import_path}")

    try:
        result = manager.import_identity(import_path=args.import_path, merge=args.merge)

        if result.success:
            print("✅ Identity imported successfully!")
            print(f"🏷️  Agent: {result.imported_agent_id}")
            print(f"📊 Imported: {result.imported_tokens} tokens")
            print(f"🗄️  Archives: {result.imported_archives}")
            print(f"🔒 Lockpoints: {result.imported_lockpoints}")

            if result.chain_integrity_verified:
                print("🔐 Chain integrity verified ✓")
            else:
                print("⚠️  Chain integrity verification failed")

            if result.conflicts_detected:
                print(f"⚠️  Conflicts detected: {len(result.conflicts_detected)}")
                for conflict in result.conflicts_detected:
                    print(f"   - {conflict}")
        else:
            print("❌ Import failed:")
            for error in result.error_messages:
                print(f"   - {error}")

    except Exception as e:
        print(f"❌ Import failed: {e}")


def cmd_verify_integrity(args):
    """Verify PMM integrity."""
    manager = create_manager(args.model_path, args.next_stage)

    if not args.next_stage:
        print("❌ Integrity verification requires next-stage features (--next-stage)")
        return

    print("🔍 Verifying PMM integrity...")

    result = manager.verify_integrity()

    if "error" in result:
        print(f"❌ Verification failed: {result['error']}")
        return

    print(f"🔐 Chain Valid: {'✅' if result['chain_valid'] else '❌'}")

    if result["chain_errors"]:
        print("⚠️  Chain Errors:")
        for error in result["chain_errors"]:
            print(f"   - {error}")

    print(
        f"🔒 Lockpoints: {result['lockpoints_valid']}/{len(result['lockpoints'])} valid"
    )

    if result["anomalies"]:
        print(f"⚠️  Anomalies detected: {len(result['anomalies'])}")
        for anomaly in result["anomalies"]:
            print(f"   - {anomaly['type']}: {anomaly['description']}")

    diagnostics = result["chain_diagnostics"]
    print("\n📊 Chain Statistics:")
    print(f"   Length: {diagnostics['chain_length']}")
    print(f"   Genesis: {diagnostics['genesis_hash'][:16]}...")
    print(f"   Avg Amplitude: {diagnostics['amplitude_stats'].get('avg', 0):.3f}")


def cmd_stats(args):
    """Show comprehensive PMM statistics."""
    manager = create_manager(args.model_path, args.next_stage)

    stats = manager.get_comprehensive_stats()

    print(f"📊 PMM Statistics for {stats['agent_name']} ({stats['agent_id']})")
    print("=" * 60)

    # Traditional stats
    trad = stats["traditional"]
    print("📝 Traditional PMM:")
    print(f"   Events: {trad['events']}")
    print(f"   Thoughts: {trad['thoughts']}")
    print(f"   Insights: {trad['insights']}")
    print(f"   Commitments: {trad['commitments']}")

    # Next-stage stats
    if args.next_stage and "next_stage" in stats:
        ns = stats["next_stage"]
        print("\n🚀 Next-Stage Features:")
        print(f"   Memory Tokens: {ns['memory_tokens']}")
        print(f"   Active Tokens: {ns['active_tokens']}")
        print(f"   Archives: {ns['archives']}")
        print(f"   Lockpoints: {ns['lockpoints']}")

        if "engines" in stats:
            engines = stats["engines"]
            print("\n🔧 Engine Status:")
            for engine, enabled in engines.items():
                status = "✅" if enabled else "❌"
                print(f"   {engine.title()}: {status}")


def cmd_quantum_analysis(args):
    """Analyze quantum memory coherence."""
    manager = create_manager(args.model_path, args.next_stage)

    if not args.next_stage:
        print("❌ Quantum analysis requires next-stage features (--next-stage)")
        return

    if not hasattr(manager, "quantum_manager"):
        print("❌ Quantum memory manager not available")
        return

    print("🌀 Analyzing quantum memory coherence...")

    coherence = manager.get_quantum_coherence()

    if "error" in coherence:
        print(f"❌ Analysis failed: {coherence['error']}")
        return

    print(f"🧠 Average Coherence: {coherence['avg_coherence']:.3f}")
    print(f"🔗 Memory Clusters: {coherence['cluster_count']}")

    if args.verbose:
        print("\n🌐 Cluster Details:")
        for cluster_id, token_ids in coherence["memory_clusters"].items():
            print(f"   {cluster_id}: {len(token_ids)} tokens")


def cmd_archive_status(args):
    """Show archive status and trigger manual archival."""
    manager = create_manager(args.model_path, args.next_stage)

    if not args.next_stage:
        print("❌ Archive management requires next-stage features (--next-stage)")
        return

    stats = manager.get_comprehensive_stats()

    if "next_stage" not in stats:
        print("❌ Next-stage statistics not available")
        return

    ns = stats["next_stage"]
    archived_tokens = ns["memory_tokens"] - ns["active_tokens"]

    print("🗄️  Archive Status:")
    print(f"   Active Tokens: {ns['active_tokens']}")
    print(f"   Archived Tokens: {archived_tokens}")
    print(f"   Archives: {ns['archives']}")

    if args.trigger_archival:
        print("\n📦 Triggering manual archival...")
        result = manager.trigger_archival()

        if "error" in result:
            print(f"❌ Archival failed: {result['error']}")
        elif "message" in result:
            print(f"ℹ️  {result['message']}")
        else:
            print(
                f"✅ Archived {result['archived_tokens']} tokens in {result['clusters_created']} clusters"
            )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PMM Next-Stage CLI - Manage next-stage PMM features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add an event with tokenization
  python pmm_cli.py add-event "Recorded an insight about memory integrity" --next-stage
  
  # Recall memories semantically
  python pmm_cli.py recall "memory integrity" --max-results 5 --next-stage
  
  # Generate text locally
  python pmm_cli.py generate "Summarize my memory state and recent insights" --next-stage
  
  # Export complete identity
  python pmm_cli.py export-identity ./my_identity --next-stage
  
  # Verify cryptographic integrity
  python pmm_cli.py verify-integrity --next-stage
        """,
    )

    parser.add_argument(
        "--model-path",
        "-m",
        default="enhanced_pmm_model.json",
        help="Path to PMM model file",
    )

    parser.add_argument(
        "--next-stage",
        "-n",
        action="store_true",
        help="Enable next-stage features (tokenization, recall, etc.)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add Event
    add_event_parser = subparsers.add_parser("add-event", help="Add event to PMM")
    add_event_parser.add_argument("summary", help="Event summary")
    add_event_parser.add_argument("--type", default="experience", help="Event type")
    add_event_parser.set_defaults(func=cmd_add_event)

    # Add Thought
    add_thought_parser = subparsers.add_parser("add-thought", help="Add thought to PMM")
    add_thought_parser.add_argument("content", help="Thought content")
    add_thought_parser.add_argument("--trigger", help="Thought trigger")
    add_thought_parser.set_defaults(func=cmd_add_thought)

    # Add Insight
    add_insight_parser = subparsers.add_parser("add-insight", help="Add insight to PMM")
    add_insight_parser.add_argument("content", help="Insight content")
    add_insight_parser.set_defaults(func=cmd_add_insight)

    # Recall Memories
    recall_parser = subparsers.add_parser(
        "recall", help="Recall memories by semantic cue"
    )
    recall_parser.add_argument("cue", help="Memory recall cue")
    recall_parser.add_argument("--max-results", type=int, default=5, help="Max results")
    recall_parser.set_defaults(func=cmd_recall_memories)

    # Generate Text
    generate_parser = subparsers.add_parser("generate", help="Generate text locally")
    generate_parser.add_argument("prompt", help="Generation prompt")
    generate_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature"
    )
    generate_parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Max tokens"
    )
    generate_parser.set_defaults(func=cmd_generate_local)

    # Export Identity
    export_parser = subparsers.add_parser("export-identity", help="Export PMM identity")
    export_parser.add_argument("export_path", help="Export directory path")
    export_parser.add_argument(
        "--no-archives",
        dest="include_archives",
        action="store_false",
        help="Exclude archives",
    )
    export_parser.set_defaults(func=cmd_export_identity)

    # Import Identity
    import_parser = subparsers.add_parser("import-identity", help="Import PMM identity")
    import_parser.add_argument("import_path", help="Import file/directory path")
    import_parser.add_argument(
        "--merge", action="store_true", help="Merge with existing identity"
    )
    import_parser.set_defaults(func=cmd_import_identity)

    # Verify Integrity
    verify_parser = subparsers.add_parser(
        "verify-integrity", help="Verify PMM integrity"
    )
    verify_parser.set_defaults(func=cmd_verify_integrity)

    # Statistics
    stats_parser = subparsers.add_parser("stats", help="Show PMM statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Quantum Analysis
    quantum_parser = subparsers.add_parser("quantum", help="Quantum memory analysis")
    quantum_parser.set_defaults(func=cmd_quantum_analysis)

    # Archive Management
    archive_parser = subparsers.add_parser("archive", help="Archive management")
    archive_parser.add_argument(
        "--trigger",
        dest="trigger_archival",
        action="store_true",
        help="Trigger manual archival",
    )
    archive_parser.set_defaults(func=cmd_archive_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
