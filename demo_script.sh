#!/bin/bash

# PMM Demo Recording Script
# This script provides a structured demo of the Persistent Mind Model
# showcasing personality evolution, commitment tracking, and model-agnostic transfer

echo "🧠 PERSISTENT MIND MODEL - LIVE DEMO"
echo "===================================="
echo ""

# Set up clean demo environment
echo "📋 Setting up demo environment..."
cd /home/scott/Documents/Projects/Business-Development/persistent-mind-model
source .venv/bin/activate

# Show initial agent status
echo ""
echo "🤖 Initial Agent Status:"
python cli.py status 2>/dev/null || echo "Creating new demo agent..."

# Demonstrate personality evolution through duel
echo ""
echo "⚔️  Running Agent Personality Duel (showing real-time trait evolution)..."
echo "This demonstrates measurable personality drift with commitment tracking:"
python duel.py

# Show post-evolution status
echo ""
echo "📊 Post-Evolution Agent Status:"
python cli.py status

# Demonstrate model-agnostic consciousness transfer
echo ""
echo "🔄 Demonstrating Model-Agnostic Consciousness Transfer..."
echo "Same personality, different LLM backends:"
python demo_model_agnostic.py

# Show reflection capabilities
echo ""
echo "🪞 Agent Self-Reflection Capabilities:"
python cli.py reflect-if-due

echo ""
echo "✨ Demo Complete! This showcases:"
echo "   • Measurable personality trait evolution"
echo "   • Commitment extraction and tracking"
echo "   • Model-agnostic consciousness transfer"
echo "   • Self-reflective meta-cognition"
echo ""
echo "🎯 Key Innovation: Model-agnostic AI consciousness with persistent personality"
