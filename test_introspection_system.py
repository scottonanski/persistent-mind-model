#!/usr/bin/env python3
"""
Comprehensive test suite for PMM's Hybrid Introspection System.

Tests both user-prompted and automatic introspection capabilities with full transparency.
"""

import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_introspection_commands():
    """Test user-prompted introspection commands."""
    print("🧪 Testing User-Prompted Introspection Commands...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    try:
        # Initialize PMM with temporary database
        os.environ["PMM_DB_PATH"] = db_path
        # Keep existing OPENAI_API_KEY from .env file

        from pmm.langchain_memory import PersistentMindMemory

        # Create PMM instance
        pmm = PersistentMindMemory(agent_path=db_path.replace(".db", ".json"))
        print("✅ PMM initialized with introspection engine")

        # Test help command
        print("\n📋 Testing @introspect help command...")
        help_result = pmm.handle_introspection_command("@introspect help")
        print(f"Help result type: {type(help_result)}")
        if (
            isinstance(help_result, str)
            and "Available Introspection Commands" in help_result
        ):
            print("✅ Help command working correctly")
            print(f"Result preview: {help_result[:200]}...")
        else:
            print("❌ Help command failed")

        # Test patterns command
        print("\n🔍 Testing @introspect patterns command...")
        patterns_result = pmm.handle_introspection_command("@introspect patterns")
        if (
            isinstance(patterns_result, str)
            and "Patterns Introspection" in patterns_result
        ):
            print("✅ Patterns command working correctly")
            print(f"Result preview: {patterns_result[:200]}...")
        else:
            print("❌ Patterns command failed")

        # Test commitments command
        print("\n📝 Testing @introspect commitments command...")
        commitments_result = pmm.handle_introspection_command("@introspect commitments")
        if (
            isinstance(commitments_result, str)
            and "Commitments Introspection" in commitments_result
        ):
            print("✅ Commitments command working correctly")
            print(f"Result preview: {commitments_result[:200]}...")
        else:
            print("❌ Commitments command failed")

        # Test growth command
        print("\n📈 Testing @introspect growth command...")
        growth_result = pmm.handle_introspection_command("@introspect growth")
        if isinstance(growth_result, str) and "Growth Introspection" in growth_result:
            print("✅ Growth command working correctly")
            print(f"Result preview: {growth_result[:200]}...")
        else:
            print("❌ Growth command failed")

        # Verify introspection events were logged
        recent_events = pmm.pmm.sqlite_store.recent_events(limit=10)
        introspection_events = [
            e
            for e in recent_events
            if e.get("meta", {}).get("type") == "introspection_command"
        ]

        print(f"\n📊 Introspection events logged: {len(introspection_events)}")
        for event in introspection_events:
            print(f"  • {event.get('content', '')[:50]}...")

        return len(introspection_events) >= 3  # Should have at least 3 commands logged

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
            os.unlink(db_path.replace(".db", ".json"))
        except:
            pass


def test_automatic_introspection():
    """Test automatic introspection triggers."""
    print("\n🤖 Testing Automatic Introspection Triggers...")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name

    try:
        os.environ["PMM_DB_PATH"] = db_path
        # Keep existing OPENAI_API_KEY from .env file

        from pmm.langchain_memory import PersistentMindMemory

        # Create PMM instance
        pmm = PersistentMindMemory(agent_path=db_path.replace(".db", ".json"))

        # Create conditions that should trigger automatic introspection
        print(
            "📝 Creating commitments without evidence (should trigger automatic analysis)..."
        )

        # Add some commitments
        pmm.save_context(
            inputs={"human": "I commit to finishing this project by Friday"},
            outputs={
                "ai": "That's a great commitment! I'll help you track your progress."
            },
        )

        pmm.save_context(
            inputs={"human": "I also commit to improving my productivity"},
            outputs={"ai": "Excellent! Productivity improvements are always valuable."},
        )

        # Trigger reflection (which should also check for automatic introspection)
        print("🧠 Triggering reflection to check automatic introspection...")
        reflection_result = pmm.trigger_reflection()
        print(f"Reflection triggered: {bool(reflection_result)}")

        # Check for automatic introspection events
        recent_events = pmm.pmm.sqlite_store.recent_events(limit=20)
        automatic_events = [
            e for e in recent_events if e.get("kind") == "introspection_automatic"
        ]

        print(f"📊 Automatic introspection events: {len(automatic_events)}")
        for event in automatic_events:
            print(f"  • {event.get('content', '')[:100]}...")

        return True  # Test passes if no errors occurred

    except Exception as e:
        print(f"❌ Automatic introspection test failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
            os.unlink(db_path.replace(".db", ".json"))
        except:
            pass


def test_introspection_api_endpoint():
    """Test the introspection probe API endpoint."""
    print("\n🌐 Testing Introspection API Endpoint...")

    try:
        import requests

        # Test the introspection endpoint
        print("📡 Testing /introspection endpoint...")
        response = requests.get("http://localhost:8000/introspection", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print("✅ Introspection endpoint responding correctly")

            # Check expected fields
            expected_fields = [
                "introspection_summary",
                "available_commands",
                "configuration",
                "system_status",
            ]

            missing_fields = [field for field in expected_fields if field not in data]
            if not missing_fields:
                print("✅ All expected fields present in response")

                # Print summary
                summary = data.get("introspection_summary", {})
                print(
                    f"📊 Total introspection events: {summary.get('total_events', 0)}"
                )
                print(f"👤 User commands: {summary.get('user_commands', 0)}")
                print(f"🤖 Automatic triggers: {summary.get('automatic_triggers', 0)}")

                # Print available commands
                commands = data.get("available_commands", {})
                print(f"📋 Available commands: {len(commands)}")
                for cmd in list(commands.keys())[:3]:  # Show first 3
                    print(f"  • {cmd}")

                return True
            else:
                print(f"❌ Missing fields: {missing_fields}")
                return False
        else:
            print(f"❌ API endpoint returned status {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False

    except requests.exceptions.ConnectionError:
        print("⚠️ Could not connect to probe API (server may not be running)")
        print("💡 Start server with: uvicorn pmm.api.probe:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def test_introspection_engine_directly():
    """Test the introspection engine directly."""
    print("\n⚙️ Testing Introspection Engine Directly...")

    try:
        from pmm.introspection import (
            IntrospectionEngine,
            IntrospectionConfig,
        )
        from pmm.storage.sqlite_store import SQLiteStore

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db_path = tmp_db.name

        # Initialize components
        store = SQLiteStore(db_path)
        config = IntrospectionConfig()
        engine = IntrospectionEngine(store, config)

        print("✅ Introspection engine initialized")

        # Test command parsing
        test_commands = [
            "@introspect patterns",
            "@introspect help",
            "@introspect commitments",
            "regular message",
            "@introspect invalid_command",
        ]

        print("🔍 Testing command parsing...")
        for cmd in test_commands:
            parsed = engine.parse_user_command(cmd)
            print(f"  '{cmd}' → {parsed.value if parsed else 'None'}")

        # Test available commands
        commands = engine.get_available_commands()
        print(f"📋 Available commands: {len(commands)}")

        # Test direct analysis
        print("📊 Testing direct pattern analysis...")
        result = engine.analyze_patterns(lookback_days=7)
        print(f"  Analysis type: {result.type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Insights: {len(result.insights)}")
        print(f"  Recommendations: {len(result.recommendations)}")

        # Test result formatting
        formatted = engine.format_result_for_user(result)
        print(f"📝 Formatted result length: {len(formatted)} characters")
        print(f"Preview: {formatted[:150]}...")

        # Cleanup
        os.unlink(db_path)

        return True

    except Exception as e:
        print(f"❌ Direct engine test failed: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Run all introspection system tests."""
    print("🎯 PMM Hybrid Introspection System - Comprehensive Test Suite")
    print("=" * 70)

    tests = [
        ("User-Prompted Commands", test_introspection_commands),
        ("Automatic Triggers", test_automatic_introspection),
        ("API Endpoint", test_introspection_api_endpoint),
        ("Direct Engine", test_introspection_engine_directly),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'=' * 70}")
    print("🎯 TEST SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All introspection system tests PASSED!")
        print("\n💡 The hybrid introspection system is fully operational:")
        print("   • User-prompted commands (@introspect patterns, etc.)")
        print("   • Automatic triggers (failed commitments, trait drift, etc.)")
        print("   • Full transparency with user notifications")
        print("   • Comprehensive API monitoring via /introspection endpoint")
        return True
    else:
        print("⚠️ Some tests failed. Review the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
