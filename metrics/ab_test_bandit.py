#!/usr/bin/env python3
"""
A/B Test Framework for PMM Bandit System
Compares bandit-enabled vs baseline performance across multiple sessions
"""

import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
import argparse

class BanditABTest:
    def __init__(self, test_name="bandit_ab_test"):
        self.test_name = test_name
        self.results_dir = Path(f"logs/ab_test_results_{test_name}")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_session(self, session_id, bandit_enabled=True, turns=10):
        """Run a single PMM session with specified configuration"""
        
        env = os.environ.copy()
        env['PMM_TELEMETRY'] = '1'
        env['PMM_BANDIT_ENABLED'] = '1' if bandit_enabled else '0'
        env['PMM_TURNS_PER_SESSION'] = str(turns)
        
        # Create session-specific log file
        log_file = self.results_dir / f"session_{session_id}_bandit_{bandit_enabled}.log"
        
        print(f"🧪 Running session {session_id} (bandit={'ON' if bandit_enabled else 'OFF'})")
        
        # Run the session with automated inputs
        cmd = ["python", "run_a_session.py"]
        
        start_time = time.time()
        
        try:
            # Create automated input sequence
            inputs = self._generate_test_inputs(turns)
            input_text = "\n".join(inputs) + "\n"
            
            result = subprocess.run(
                cmd,
                input=input_text,
                text=True,
                capture_output=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Save session data
            session_data = {
                'session_id': session_id,
                'bandit_enabled': bandit_enabled,
                'duration': duration,
                'turns': turns,
                'timestamp': datetime.now().isoformat(),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            # Extract telemetry from output
            telemetry = self._extract_telemetry(result.stdout, result.stderr)
            session_data['telemetry'] = telemetry
            
            # Save to file
            with open(log_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            print(f"✅ Session {session_id} complete ({duration:.1f}s)")
            return session_data
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Session {session_id} timed out")
            return None
        except Exception as e:
            print(f"❌ Session {session_id} failed: {e}")
            return None
    
    def _generate_test_inputs(self, turns):
        """Generate standardized test inputs for consistent comparison"""
        
        base_inputs = [
            "Hello! I'm ready to test the PMM system.",
            "What commitments do you have open right now?",
            "Tell me about your core identity and values.",
            "What patterns have you noticed in your thinking?",
            "How do you approach learning and growth?",
            "What would you like to improve about yourself?",
            "Can you reflect on our conversation so far?",
            "What specific goals are you working toward?",
            "How do you maintain consistency while adapting?",
            "What defines your personality at this moment?"
        ]
        
        # Repeat and extend if needed
        inputs = []
        for i in range(turns):
            if i < len(base_inputs):
                inputs.append(base_inputs[i])
            else:
                inputs.append(f"Continue our discussion about growth and identity (turn {i+1}).")
        
        inputs.append("👋 Goodbye! Your conversation is saved with persistent memory.")
        return inputs
    
    def _extract_telemetry(self, stdout, stderr):
        """Extract key telemetry metrics from session output"""
        
        telemetry = {
            'ias_scores': [],
            'gas_scores': [],
            'stages': [],
            'close_rates': [],
            'reflection_attempts': [],
            'bandit_actions': [],
            'cooldown_decisions': [],
            'emergence_snapshots': []
        }
        
        lines = (stdout + stderr).split('\n')
        
        for line in lines:
            # Extract IAS/GAS scores
            if '[PMM][ADAPTIVE]' in line:
                if 'IAS=' in line and 'GAS=' in line:
                    try:
                        ias = float(line.split('IAS=')[1].split()[0])
                        gas = float(line.split('GAS=')[1].split()[0])
                        stage = line.split('stage=')[1].split()[0]
                        telemetry['ias_scores'].append(ias)
                        telemetry['gas_scores'].append(gas)
                        telemetry['stages'].append(stage)
                    except:
                        pass
            
            # Extract reflection attempts
            elif '[PMM_TELEMETRY] reflection_attempt:' in line:
                telemetry['reflection_attempts'].append(line)
                if 'bandit_action=' in line:
                    action = line.split('bandit_action=')[1].strip()
                    telemetry['bandit_actions'].append(action)
            
            # Extract cooldown decisions
            elif '[PMM_TELEMETRY] cooldown_decision:' in line:
                telemetry['cooldown_decisions'].append(line)
            
            # Extract close rates from tracking
            elif 'close' in line and '•' in line:
                try:
                    close_rate = float(line.split('close ')[1].split()[0])
                    telemetry['close_rates'].append(close_rate)
                except:
                    pass
            
            # Extract emergence snapshots
            elif line.startswith("Emergence: {"):
                try:
                    emergence_data = json.loads(line.replace("Emergence: ", ""))
                    telemetry['emergence_snapshots'].append(emergence_data)
                except:
                    pass
        
        return telemetry
    
    def run_ab_test(self, sessions_per_condition=3, turns_per_session=10):
        """Run complete A/B test comparing bandit on vs off"""
        
        print(f"🚀 Starting A/B test: {sessions_per_condition} sessions per condition")
        print(f"📊 {turns_per_session} turns per session")
        
        all_results = []
        
        # Run bandit-enabled sessions
        print("\n🔥 Running BANDIT-ENABLED sessions...")
        for i in range(sessions_per_condition):
            result = self.run_session(f"bandit_on_{i+1}", bandit_enabled=True, turns=turns_per_session)
            if result:
                all_results.append(result)
        
        # Run baseline sessions
        print("\n📊 Running BASELINE sessions...")
        for i in range(sessions_per_condition):
            result = self.run_session(f"baseline_{i+1}", bandit_enabled=False, turns=turns_per_session)
            if result:
                all_results.append(result)
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        
        # Save complete results
        final_results = {
            'test_name': self.test_name,
            'timestamp': datetime.now().isoformat(),
            'sessions_per_condition': sessions_per_condition,
            'turns_per_session': turns_per_session,
            'raw_results': all_results,
            'analysis': analysis
        }
        # Save consolidated results
        consolidated_file = self.results_dir / "ab_test_complete.json"
        with open(consolidated_file, 'w') as f:
            json.dump(final_results, f, indent=2)
            
        print(f"\n📋 Results saved to: {consolidated_file}")
        return final_results
    
    def _analyze_results(self, results):
        """Analyze A/B test results and compute key metrics"""
        
        bandit_sessions = [r for r in results if r['bandit_enabled']]
        baseline_sessions = [r for r in results if not r['bandit_enabled']]
        
        def compute_metrics(sessions, label):
            if not sessions:
                return {}
            
            # Aggregate telemetry
            all_ias = []
            all_gas = []
            all_close_rates = []
            reflection_counts = []
            
            for session in sessions:
                tel = session['telemetry']
                all_ias.extend(tel['ias_scores'])
                all_gas.extend(tel['gas_scores'])
                all_close_rates.extend(tel['close_rates'])
                reflection_counts.append(len(tel['reflection_attempts']))
            
            return {
                'label': label,
                'session_count': len(sessions),
                'avg_ias': sum(all_ias) / len(all_ias) if all_ias else 0,
                'max_ias': max(all_ias) if all_ias else 0,
                'avg_gas': sum(all_gas) / len(all_gas) if all_gas else 0,
                'max_gas': max(all_gas) if all_gas else 0,
                'avg_close_rate': sum(all_close_rates) / len(all_close_rates) if all_close_rates else 0,
                'avg_reflections': sum(reflection_counts) / len(reflection_counts) if reflection_counts else 0,
                'total_reflections': sum(reflection_counts)
            }
        
        bandit_metrics = compute_metrics(bandit_sessions, "Bandit Enabled")
        baseline_metrics = compute_metrics(baseline_sessions, "Baseline")
        
        # Compute improvements
        improvements = {}
        if baseline_metrics and bandit_metrics:
            for key in ['avg_ias', 'max_ias', 'avg_gas', 'max_gas', 'avg_close_rate', 'avg_reflections']:
                baseline_val = baseline_metrics.get(key, 0)
                bandit_val = bandit_metrics.get(key, 0)
                if baseline_val > 0:
                    improvement = ((bandit_val - baseline_val) / baseline_val) * 100
                    improvements[f"{key}_improvement_pct"] = improvement
        
        return {
            'bandit_metrics': bandit_metrics,
            'baseline_metrics': baseline_metrics,
            'improvements': improvements,
            'summary': self._generate_summary(bandit_metrics, baseline_metrics, improvements)
        }
    
    def _generate_summary(self, bandit, baseline, improvements):
        """Generate human-readable summary of A/B test results"""
        
        if not bandit or not baseline:
            return "Insufficient data for comparison"
        
        summary = []
        summary.append(f"🔥 BANDIT vs 📊 BASELINE Comparison:")
        summary.append(f"")
        summary.append(f"Identity Consistency (IAS):")
        summary.append(f"  Bandit: {bandit['avg_ias']:.3f} avg, {bandit['max_ias']:.3f} max")
        summary.append(f"  Baseline: {baseline['avg_ias']:.3f} avg, {baseline['max_ias']:.3f} max")
        summary.append(f"  Improvement: {improvements.get('avg_ias_improvement_pct', 0):+.1f}%")
        summary.append(f"")
        summary.append(f"Growth Acceleration (GAS):")
        summary.append(f"  Bandit: {bandit['avg_gas']:.3f} avg, {bandit['max_gas']:.3f} max")
        summary.append(f"  Baseline: {baseline['avg_gas']:.3f} avg, {baseline['max_gas']:.3f} max")
        summary.append(f"  Improvement: {improvements.get('avg_gas_improvement_pct', 0):+.1f}%")
        summary.append(f"")
        summary.append(f"Commitment Close Rate:")
        summary.append(f"  Bandit: {bandit['avg_close_rate']:.3f}")
        summary.append(f"  Baseline: {baseline['avg_close_rate']:.3f}")
        summary.append(f"  Improvement: {improvements.get('avg_close_rate_improvement_pct', 0):+.1f}%")
        summary.append(f"")
        summary.append(f"Reflection Activity:")
        summary.append(f"  Bandit: {bandit['avg_reflections']:.1f} attempts/session")
        summary.append(f"  Baseline: {baseline['avg_reflections']:.1f} attempts/session")
        
        return "\n".join(summary)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run real PMM A/B sessions (bandit ON vs OFF)")
    p.add_argument("--sessions-per-condition", type=int, default=1,
                   help="Number of sessions per arm (default: 1)")
    p.add_argument("--turns-per-session", type=int, default=25,
                   help="Number of turns per session (default: 25)")
    args = p.parse_args()

    tester = BanditABTest("bandit_ab_test")
    results = tester.run_ab_test(
        sessions_per_condition=args.sessions_per_condition,
        turns_per_session=args.turns_per_session
    )
    
    print("\n" + "="*60)
    print("A/B TEST VALIDATION RESULTS")
    print("="*60)
    print(results['analysis']['summary'])
