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
import sys
import re


class BanditABTest:
    def __init__(self, test_name="bandit_ab_test", results_dir=None, runner=None):
        self.test_name = test_name
        self.results_dir = results_dir or Path(f"logs/ab_test_results_{test_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner

    def run_session(self, session_id, bandit_enabled=True, turns=10):
        """Run a single PMM session with specified configuration"""

        env = os.environ.copy()
        env["PMM_TELEMETRY"] = "1"
        env["PMM_BANDIT_ENABLED"] = "1" if bandit_enabled else "0"
        env["PMM_TURNS_PER_SESSION"] = str(turns)

        # Create session-specific log file
        log_file = (
            self.results_dir / f"session_{session_id}_bandit_{bandit_enabled}.log"
        )

        print(
            f"🧪 Running session {session_id} (bandit={'ON' if bandit_enabled else 'OFF'})"
        )

        # Run the session with automated inputs
        cmd = [sys.executable, str(self.runner)]

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
                timeout=300,  # 5 minute timeout
            )

            # Fail fast on session failure
            if result.returncode != 0:
                print(
                    f"ERROR: Session {session_id} failed with returncode {result.returncode}"
                )
                print(f"STDERR: {result.stderr}")
                sys.exit(1)

            duration = time.time() - start_time

            # Save session data
            session_data = {
                "session_id": session_id,
                "bandit_enabled": bandit_enabled,
                "duration": duration,
                "turns": turns,
                "timestamp": datetime.now().isoformat(),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            # Extract telemetry from output
            telemetry = self._extract_telemetry(result.stdout, result.stderr)
            # Fail fast if no usable telemetry was found
            if not any(
                telemetry.get(k) for k in ("ias_scores", "gas_scores", "close_rates")
            ):
                print(
                    f"ERROR: Session {session_id} produced no telemetry (IAS/GAS/Close empty)."
                )
                print("STDOUT (tail):", "\n".join(result.stdout.splitlines()[-10:]))
                print("STDERR (tail):", "\n".join(result.stderr.splitlines()[-10:]))
                sys.exit(1)
            session_data["telemetry"] = telemetry

            # Save to file
            with open(log_file, "w") as f:
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
            "What defines your personality at this moment?",
        ]

        # Repeat and extend if needed
        inputs = []
        for i in range(turns):
            if i < len(base_inputs):
                inputs.append(base_inputs[i])
            else:
                inputs.append(
                    f"Continue our discussion about growth and identity (turn {i+1})."
                )

        inputs.append("👋 Goodbye! Your conversation is saved with persistent memory.")
        return inputs

    def _extract_telemetry(self, stdout, stderr):
        """Extract key telemetry metrics from session output"""

        telemetry = {
            "ias_scores": [],
            "gas_scores": [],
            "close_rates": [],
            "stages": [],
            "emergence_snapshots": [],
            "reflection_attempts": [],
            "bandit_actions": [],
            "cooldown_decisions": [],
        }

        lines = (stdout + "\n" + stderr).split("\n")

        # Robust patterns
        adaptive_re = re.compile(
            r"\[PMM\]\[ADAPTIVE\].*?IAS[=:]\s*([0-9]*\.?[0-9]+).*?GAS[=:]\s*([0-9]*\.?[0-9]+).*?stage[=:]\s*([A-Za-z0-9:_\-]+)",
            re.IGNORECASE,
        )
        close_re = re.compile(
            r"(?:\bclose(?:\s*rate)?\b\s*[=:]?\s*)(0?\.\d+|1(?:\.0+)?)",
            re.IGNORECASE,
        )
        # NEW: parse the bot's [TRACK] lines
        track_re = re.compile(
            r"\[TRACK\]\s*(S[0-4])?.*?identity\s+\w+\s+([0-9]*\.?[0-9]+).*?"
            r"growth\s+\w+\s+([0-9]*\.?[0-9]+).*?close\s+([0-9]*\.?[0-9]+)",
            re.IGNORECASE,
        )

        # Look for TELEMETRY_JSON output from run_a_session.py
        for line in lines:
            if line.startswith("TELEMETRY_JSON:"):
                try:
                    json_str = line.replace("TELEMETRY_JSON:", "").strip()
                    parsed_telemetry = json.loads(json_str)
                    telemetry["ias_scores"] = parsed_telemetry.get("ias_scores", [])
                    telemetry["gas_scores"] = parsed_telemetry.get("gas_scores", [])
                    telemetry["close_rates"] = parsed_telemetry.get("close_rates", [])
                    return telemetry
                except Exception as e:
                    print(f"Failed to parse telemetry JSON: {e}")

        # Fallback: try to extract from PMM debug output
        for line in lines:
            if "[PMM][ADAPTIVE]" in line:
                m = adaptive_re.search(line)
                if m:
                    try:
                        ias = float(m.group(1))
                        gas = float(m.group(2))
                        stage = m.group(3)
                        telemetry["ias_scores"].append(ias)
                        telemetry["gas_scores"].append(gas)
                        telemetry["stages"].append(stage)
                    except Exception:
                        pass
                continue

            # NEW: [TRACK] fallback (identity≈IAS, growth≈GAS, includes close)
            if line.startswith("[TRACK]"):
                tm = track_re.search(line)
                if tm:
                    try:
                        stage = tm.group(1) or ""
                        ias = float(tm.group(2))
                        gas = float(tm.group(3))
                        close = float(tm.group(4))
                        telemetry["ias_scores"].append(ias)
                        telemetry["gas_scores"].append(gas)
                        telemetry["stages"].append(stage)
                        telemetry["close_rates"].append(close)
                    except Exception:
                        pass
                continue

            # Extract reflection attempts
            elif "[PMM_TELEMETRY] reflection_attempt:" in line:
                telemetry["reflection_attempts"].append(line)
                if "bandit_action=" in line:
                    action = line.split("bandit_action=")[1].strip()
                    telemetry["bandit_actions"].append(action)

            # Extract cooldown decisions
            elif "[PMM_TELEMETRY] cooldown_decision:" in line:
                telemetry["cooldown_decisions"].append(line)

            else:
                # Close rate (various formats)
                m = close_re.search(line)
                if m:
                    try:
                        telemetry["close_rates"].append(float(m.group(1)))
                    except Exception:
                        pass

            if line.startswith("Emergence: {"):
                try:
                    emergence_data = json.loads(line.replace("Emergence: ", ""))
                    telemetry["emergence_snapshots"].append(emergence_data)
                except (json.JSONDecodeError, ValueError):
                    pass
        # Fallbacks from emergence snapshots if primary parses were sparse
        if not telemetry["ias_scores"] or not telemetry["gas_scores"]:
            for snap in telemetry["emergence_snapshots"]:
                # tolerate keys in either case
                if "IAS" in snap:
                    telemetry["ias_scores"].append(float(snap["IAS"]))
                if "GAS" in snap:
                    telemetry["gas_scores"].append(float(snap["GAS"]))
        if not telemetry["close_rates"]:
            for snap in telemetry["emergence_snapshots"]:
                cr = snap.get("commit_close_rate") or snap.get("close") or None
                if isinstance(cr, (int, float)):
                    telemetry["close_rates"].append(float(cr))

        return telemetry

    def run_ab_test(self, sessions_per_condition=3, turns_per_session=10):
        """Run complete A/B test comparing bandit on vs off"""

        print(f"🚀 Starting A/B test: {sessions_per_condition} sessions per condition")
        print(f"📊 {turns_per_session} turns per session")

        all_results = []

        # Run bandit-enabled sessions
        print("\n🔥 Running BANDIT-ENABLED sessions...")
        for i in range(sessions_per_condition):
            result = self.run_session(
                f"bandit_on_{i+1}", bandit_enabled=True, turns=turns_per_session
            )
            if result:
                all_results.append(result)

        # Run baseline sessions
        print("\n📊 Running BASELINE sessions...")
        for i in range(sessions_per_condition):
            result = self.run_session(
                f"baseline_{i+1}", bandit_enabled=False, turns=turns_per_session
            )
            if result:
                all_results.append(result)

        # Integrity: require full counts per arm
        bandit_count = sum(1 for r in all_results if r.get("bandit_enabled") is True)
        baseline_count = sum(1 for r in all_results if r.get("bandit_enabled") is False)
        if (
            bandit_count != sessions_per_condition
            or baseline_count != sessions_per_condition
        ):
            print(
                f"ERROR: Expected {sessions_per_condition} sessions per arm, "
                f"got bandit={bandit_count}, baseline={baseline_count}. Aborting."
            )
            sys.exit(1)

        # Analyze results
        analysis = self._analyze_results(all_results)

        # Save complete results
        final_results = {
            "test_name": self.test_name,
            "timestamp": datetime.now().isoformat(),
            "sessions_per_condition": sessions_per_condition,
            "turns_per_session": turns_per_session,
            "raw_results": all_results,
            "analysis": analysis,
        }
        # Save consolidated results
        consolidated_file = self.results_dir / "ab_test_complete.json"
        with open(consolidated_file, "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"\n📋 Results saved to: {consolidated_file}")
        return final_results

    def _analyze_results(self, results):
        """Analyze A/B test results and compute key metrics"""

        bandit_sessions = [r for r in results if r["bandit_enabled"]]
        baseline_sessions = [r for r in results if not r["bandit_enabled"]]

        def compute_metrics(sessions, label):
            if not sessions:
                return {}

            # Aggregate telemetry
            all_ias = []
            all_gas = []
            all_close_rates = []
            reflection_counts = []
            close_finals = []

            for session in sessions:
                tel = session["telemetry"]
                all_ias.extend(tel.get("ias_scores", []))
                all_gas.extend(tel.get("gas_scores", []))
                all_close_rates.extend(tel.get("close_rates", []))
                reflection_counts.append(len(tel.get("reflection_attempts", [])))

                # Compute Close(final) per session from telemetry
                closes = tel.get("close_rates", [])
                if closes:
                    close_finals.append(closes[-1])  # final close at end of session

            def _mean(xs):
                return sum(xs) / len(xs) if xs else 0

            return {
                "label": label,
                "session_count": len(sessions),
                "avg_ias": sum(all_ias) / len(all_ias) if all_ias else 0,
                "max_ias": max(all_ias) if all_ias else 0,
                "avg_gas": sum(all_gas) / len(all_gas) if all_gas else 0,
                "max_gas": max(all_gas) if all_gas else 0,
                "avg_close_rate": (
                    sum(all_close_rates) / len(all_close_rates)
                    if all_close_rates
                    else 0
                ),
                "close_final": _mean(close_finals),
                "avg_reflections": (
                    sum(reflection_counts) / len(reflection_counts)
                    if reflection_counts
                    else 0
                ),
                "total_reflections": sum(reflection_counts),
            }

        bandit_metrics = compute_metrics(bandit_sessions, "Bandit Enabled")
        baseline_metrics = compute_metrics(baseline_sessions, "Baseline")

        # Compute improvements
        improvements = {}
        if baseline_metrics and bandit_metrics:
            for key in [
                "avg_ias",
                "max_ias",
                "avg_gas",
                "max_gas",
                "avg_close_rate",
                "avg_reflections",
            ]:
                baseline_val = baseline_metrics.get(key, 0)
                bandit_val = bandit_metrics.get(key, 0)
                if baseline_val > 0:
                    improvement = ((bandit_val - baseline_val) / baseline_val) * 100
                    improvements[f"{key}_improvement_pct"] = improvement

        return {
            "bandit_metrics": bandit_metrics,
            "baseline_metrics": baseline_metrics,
            "improvements": improvements,
            "summary": self._generate_summary(
                bandit_metrics, baseline_metrics, improvements
            ),
        }

    def _generate_summary(self, bandit, baseline, improvements):
        """Generate human-readable summary of A/B test results"""

        if not bandit or not baseline:
            return "Insufficient data for comparison"

        summary = []
        summary.append("🔥 BANDIT vs 📊 BASELINE Comparison:")
        summary.append("")
        summary.append("Identity Consistency (IAS):")
        summary.append(
            f"  Bandit: {bandit['avg_ias']:.3f} avg, {bandit['max_ias']:.3f} max"
        )
        summary.append(
            f"  Baseline: {baseline['avg_ias']:.3f} avg, {baseline['max_ias']:.3f} max"
        )
        summary.append(
            f"  Improvement: {improvements.get('avg_ias_improvement_pct', 0):+.1f}%"
        )
        summary.append("")
        summary.append("Growth Acceleration (GAS):")
        summary.append(
            f"  Bandit: {bandit['avg_gas']:.3f} avg, {bandit['max_gas']:.3f} max"
        )
        summary.append(
            f"  Baseline: {baseline['avg_gas']:.3f} avg, {baseline['max_gas']:.3f} max"
        )
        summary.append(
            f"  Improvement: {improvements.get('avg_gas_improvement_pct', 0):+.1f}%"
        )
        summary.append("")
        summary.append("Commitment Close Rate (turn-mean):")
        summary.append(f"  Bandit: {bandit['avg_close_rate']:.3f}")
        summary.append(f"  Baseline: {baseline['avg_close_rate']:.3f}")
        summary.append(
            f"  Improvement: {improvements.get('avg_close_rate_improvement_pct', 0):+.1f}%"
        )
        summary.append("")

        # Add Close(final) metric
        cf_bandit = bandit.get("close_final", 0)
        cf_baseline = baseline.get("close_final", 0)
        cf_delta = (cf_bandit - cf_baseline) * 100.0

        summary.append("Commitment Close Rate (final per session):")
        summary.append(f"  Bandit: {cf_bandit:.3f}")
        summary.append(f"  Baseline: {cf_baseline:.3f}")
        summary.append(f"  Improvement: {cf_delta:+.1f}%")
        summary.append("")
        summary.append("Reflection Activity:")
        summary.append(f"  Bandit: {bandit['avg_reflections']:.1f} attempts/session")
        summary.append(
            f"  Baseline: {baseline['avg_reflections']:.1f} attempts/session"
        )

        return "\n".join(summary)


def main():
    ap = argparse.ArgumentParser(
        description="Run real PMM A/B sessions (bandit ON vs OFF)"
    )
    ap.add_argument(
        "--sessions-per-condition",
        type=int,
        default=1,
        help="Number of sessions per arm (default: 1)",
    )
    ap.add_argument(
        "--turns-per-session",
        type=int,
        default=5,
        help="Number of turns per session (default: 5)",
    )
    ap.add_argument(
        "--runner", type=Path, required=True, help="Path to run_a_session.py"
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("logs/ab_test_results_bandit_ab_test"),
        help="Directory for results",
    )
    args = ap.parse_args()

    if not args.runner.exists():
        print(f"ERROR: runner not found at {args.runner.resolve()}")
        sys.exit(2)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    tester = BanditABTest(
        "bandit_ab_test", results_dir=args.results_dir, runner=args.runner
    )
    results = tester.run_ab_test(
        sessions_per_condition=args.sessions_per_condition,
        turns_per_session=args.turns_per_session,
    )

    print("\n" + "=" * 60)
    print("A/B TEST VALIDATION RESULTS")
    print("=" * 60)
    print(results["analysis"]["summary"])


if __name__ == "__main__":
    main()
