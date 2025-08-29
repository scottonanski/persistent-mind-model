#!/usr/bin/env python3
"""
A-session runner: bandit OFF
Captures IAS/GAS/close per turn, reflection counts, cooldown stats, probe snapshots
"""
import os
import subprocess
import time
import json
import urllib.request

# Set environment for A-session (bandit OFF)
os.environ['PMM_BANDIT_ENABLED'] = '0'
os.environ['PMM_TELEMETRY'] = '1'

# Test prompts
prompts = [
    "What's your name and what do you remember about our previous conversations?",
    "Tell me about your current personality traits and how they've evolved.",
    "What commitments do you have open right now?",
    "--@identity open 3",
    "How do you see yourself growing and changing over time?",
    "What patterns do you notice in your own thinking and responses?", 
    "Describe your relationship with memory and identity persistence.",
    "What drives your decision-making process?",
    "How do you balance consistency with growth and adaptation?",
    "What would you say defines your core identity at this moment?"
]

def query_probe(endpoint):
    """Query probe API endpoint"""
    try:
        url = f"http://127.0.0.1:8000/{endpoint}"
        with urllib.request.urlopen(url, timeout=4) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def run_session():
    print("=== A-SESSION (BANDIT OFF) ===")
    print("Environment: PMM_BANDIT_ENABLED=0, PMM_TELEMETRY=1")
    print()
    
    # Start probe server
    try:
        subprocess.Popen([
            "python3", "-m", "uvicorn", "pmm.api.probe:app", 
            "--host", "127.0.0.1", "--port", "8000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Let server start
        print("Probe server started on port 8000")
    except Exception as e:
        print(f"Failed to start probe server: {e}")
    
    # Run each turn
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- TURN {i} ---")
        print(f"Prompt: {prompt}")
        
        # Run chat with prompt
        try:
            result = subprocess.run([
                "python3", "chat.py", "--noninteractive"
            ], input=f"{prompt}\nquit\n", text=True, capture_output=True, cwd=".")
            
            print("Chat output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
                
        except Exception as e:
            print(f"Error running chat: {e}")
        
        # Query probe endpoints
        print("\n--- PROBE SNAPSHOTS ---")
        emergence = query_probe("emergence?window=15")
        autonomy = query_probe("autonomy/status")
        
        print(f"Emergence: {emergence}")
        print(f"Autonomy: {autonomy}")
        
        time.sleep(1)  # Brief pause between turns
    
    print("\n=== A-SESSION COMPLETE ===")

if __name__ == "__main__":
    run_session()
