#!/usr/bin/env python3
"""
Script to permanently purge legacy commitments from PMM storage.
Run this to clean out old hardcoded commitments and start fresh with autonomous development focus.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from pmm.self_model_manager import SelfModelManager


def main():
    """Purge legacy commitments from PMM storage."""
    print("🧹 Starting legacy commitment purge...")

    # Initialize PMM manager
    manager = SelfModelManager("persistent_self_model.json")

    # Purge legacy commitments
    purged_count = manager.purge_legacy_commitments()

    if purged_count > 0:
        print(f"✅ Successfully purged {purged_count} legacy commitments")
        print("🎯 PMM now focuses on autonomous development directives:")
    else:
        print("✅ No legacy commitments found - PMM is already clean")

    # Show remaining open commitments
    remaining = manager.get_open_commitments()
    if remaining:
        print(f"\n📋 {len(remaining)} remaining open commitments:")
        for commit in remaining:
            print(f"   - {commit['text'][:80]}...")
    else:
        print(
            "\n📋 No remaining open commitments - clean slate for autonomous development"
        )


if __name__ == "__main__":
    main()
