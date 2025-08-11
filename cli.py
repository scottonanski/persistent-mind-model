#!/usr/bin/env python3
from dotenv import load_dotenv
import argparse
from pmm.self_model_manager import SelfModelManager
from pmm.reflection import reflect_once
from pmm.llm import OpenAIClient

# Load environment after imports
load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "validate",
            "ingest-event",
            "apply-drift",
            "reflect",
            "status",
            "reflect-if-due",
        ],
    )  # noqa: E501
    ap.add_argument("--summary", help="event summary")
    ap.add_argument(
        "--target", action="append", help="effects target path (repeatable)"
    )
    ap.add_argument("--delta", action="append", type=float)
    ap.add_argument("--conf", action="append", type=float)
    args = ap.parse_args()

    mgr = SelfModelManager()

    if args.cmd == "validate":
        # will raise if invalid
        mgr.save_model()
        print("OK: model matches schema (or validation disabled until schema present).")
        return

    if args.cmd == "ingest-event":
        effects = []
        if args.target:
            for i, t in enumerate(args.target):
                d = args.delta[i] if args.delta and i < len(args.delta) else 0.0
                c = args.conf[i] if args.conf and i < len(args.conf) else 0.0
                effects.append({"target": t, "delta": d, "confidence": c})
        ev = mgr.add_event(summary=args.summary or "unspecified", effects=effects)
        print(f"ingested {ev.id}")
        return

    if args.cmd == "apply-drift":
        net = mgr.apply_drift_and_save()
        print("applied:", net)
        return

    if args.cmd == "reflect":
        ins = reflect_once(mgr, OpenAIClient())
        if ins:
            print(ins.content)
        else:
            print("no insight")
        return

    if args.cmd == "status":
        m = mgr.model
        sk = m.self_knowledge
        print(
            {
                "id": m.core_identity.id,
                "name": m.core_identity.name,
                "events": len(sk.autobiographical_events),
                "thoughts": len(sk.thoughts),
                "insights": len(sk.insights),
                "last_reflection_at": m.metrics.last_reflection_at,
                "drift_velocity": m.metrics.drift_velocity,
            }
        )
        return

    if args.cmd == "reflect-if-due":
        from datetime import datetime, timedelta, timezone

        cadence = mgr.model.metrics.reflection_cadence_days or 7
        last = mgr.model.metrics.last_reflection_at
        ok = True
        if last:
            try:
                # Ensure timezone-aware parsing; stored timestamps end with 'Z' for UTC
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                now_utc = datetime.now(timezone.utc)
                ok = (now_utc - last_dt) >= timedelta(days=cadence)
            except Exception:
                # If parsing fails, be conservative: skip instead of forcing reflection
                ok = False
        if not ok:
            print("skip: cadence not reached")
            return
        ins = reflect_once(mgr, OpenAIClient())
        print(ins.content if ins else "no insight")
        return


if __name__ == "__main__":
    main()
