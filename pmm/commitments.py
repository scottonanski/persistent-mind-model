#!/usr/bin/env python3
"""
Commitment lifecycle management for Persistent Mind Model.
Tracks agent commitments from creation to completion.
"""

import re
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Commitment:
    """A commitment made by an agent during reflection."""
    cid: str
    text: str
    created_at: str
    source_insight_id: str
    status: str = "open"  # open, closed, expired
    due: Optional[str] = None
    closed_at: Optional[str] = None
    close_note: Optional[str] = None
    ngrams: List[str] = None  # 3-grams for matching


class CommitmentTracker:
    """Manages commitment lifecycle and completion detection."""
    
    def __init__(self):
        self.commitments: Dict[str, Commitment] = {}
    
    def extract_commitment(self, text: str) -> Tuple[Optional[str], List[str]]:
        """Extract commitment from text and return normalized sentence + 3-grams."""
        lines = text.split('.')
        for line in lines:
            line = line.strip()
            if any(starter in line.lower() for starter in ['i will', 'next:', 'i plan to', 'i commit to']):
                # Clean up the commitment text
                commitment = line.replace('Next:', '').replace('next:', '').strip()
                if commitment:
                    # Generate 3-grams for matching
                    words = commitment.lower().split()
                    ngrams = []
                    for i in range(len(words) - 2):
                        if all(len(w) > 2 for w in words[i:i+3]):  # Skip short words
                            ngrams.append(' '.join(words[i:i+3]))
                    return commitment, ngrams
        return None, []
    
    def add_commitment(
        self, 
        text: str, 
        source_insight_id: str, 
        due: Optional[str] = None
    ) -> str:
        """Add a new commitment and return its ID."""
        commitment_text, ngrams = self.extract_commitment(text)
        if not commitment_text:
            return ""
        
        cid = f"c{len(self.commitments) + 1}"
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        commitment = Commitment(
            cid=cid,
            text=commitment_text,
            created_at=ts,
            source_insight_id=source_insight_id,
            due=due,
            ngrams=ngrams or []
        )
        
        self.commitments[cid] = commitment
        return cid
    
    def mark_commitment(self, cid: str, status: str, note: Optional[str] = None) -> bool:
        """Manually mark a commitment as closed/completed."""
        if cid not in self.commitments:
            return False
        
        commitment = self.commitments[cid]
        commitment.status = status
        commitment.closed_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        commitment.close_note = note
        return True
    
    def auto_close_from_event(self, event_text: str) -> List[str]:
        """Auto-close commitments mentioned in event descriptions."""
        closed_cids = []
        event_lower = event_text.lower()
        
        for cid, commitment in self.commitments.items():
            if commitment.status != "open":
                continue
            
            # Check if event mentions commitment ID or key terms
            if cid in event_text or any(
                ngram in event_lower for ngram in commitment.ngrams[:3]  # Top 3 ngrams
            ):
                self.mark_commitment(cid, "closed", f"Auto-closed from event: {event_text[:50]}...")
                closed_cids.append(cid)
        
        return closed_cids
    
    def auto_close_from_reflection(self, reflection_text: str) -> List[str]:
        """Auto-close commitments based on reflection completion signals."""
        closed_cids = []
        reflection_lower = reflection_text.lower()
        
        # Look for completion signals
        completion_signals = ['done', 'completed', 'finished', 'accomplished', 'achieved']
        has_completion_signal = any(signal in reflection_lower for signal in completion_signals)
        
        if not has_completion_signal:
            return closed_cids
        
        for cid, commitment in self.commitments.items():
            if commitment.status != "open":
                continue
            
            # Check if reflection mentions commitment terms
            matching_ngrams = sum(1 for ngram in commitment.ngrams if ngram in reflection_lower)
            if matching_ngrams >= 2:  # At least 2 matching 3-grams
                self.mark_commitment(cid, "closed", f"Auto-closed from reflection completion")
                closed_cids.append(cid)
        
        return closed_cids
    
    def get_open_commitments(self) -> List[Dict]:
        """Get all open commitments."""
        return [
            {
                "cid": c.cid,
                "text": c.text,
                "created_at": c.created_at,
                "source_insight_id": c.source_insight_id,
                "due": c.due
            }
            for c in self.commitments.values()
            if c.status == "open"
        ]
    
    def get_commitment_metrics(self) -> Dict:
        """Calculate commitment completion metrics."""
        total = len(self.commitments)
        open_count = sum(1 for c in self.commitments.values() if c.status == "open")
        closed_count = sum(1 for c in self.commitments.values() if c.status == "closed")
        
        # Calculate median time to close for closed commitments
        close_times = []
        for c in self.commitments.values():
            if c.status == "closed" and c.closed_at:
                try:
                    created = datetime.fromisoformat(c.created_at.replace("Z", ""))
                    closed = datetime.fromisoformat(c.closed_at.replace("Z", ""))
                    close_times.append((closed - created).total_seconds() / 3600)  # hours
                except Exception:
                    continue
        
        median_time_to_close = sorted(close_times)[len(close_times)//2] if close_times else 0
        
        return {
            "commitments_total": total,
            "commitments_open": open_count,
            "commitments_closed": closed_count,
            "close_rate": closed_count / total if total > 0 else 0,
            "median_time_to_close_hours": median_time_to_close
        }
    
    def expire_old_commitments(self, days_old: int = 30) -> List[str]:
        """Mark old commitments as expired."""
        expired_cids = []
        cutoff = datetime.utcnow() - timedelta(days=days_old)
        
        for cid, commitment in self.commitments.items():
            if commitment.status != "open":
                continue
            
            try:
                created = datetime.fromisoformat(commitment.created_at.replace("Z", ""))
                if created < cutoff:
                    self.mark_commitment(cid, "expired", f"Auto-expired after {days_old} days")
                    expired_cids.append(cid)
            except Exception:
                continue
        
        return expired_cids
