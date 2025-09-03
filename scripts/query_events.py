import argparse
import sys
from pathlib import Path

# Add PMM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmm.storage.sqlite_store import SQLiteStore


def query_events(kind: str, limit: int):
    """Query the PMM event log."""
    # This assumes the default agent path and DB location.
    agent_path = Path("test_agent.json")
    db_path = agent_path.with_suffix(".db")
    store = SQLiteStore(db_path)

    # The events are stored in the `events` table.
    # We can query it directly using the internal sqlite_store connection.
    try:
        cursor = store.conn.cursor()
        if kind == "all":
            cursor.execute("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,))
        else:
            cursor.execute(
                "SELECT * FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
                (kind, limit),
            )

        rows = cursor.fetchall()
        if not rows:
            print(f"No events of kind '{kind}' found.")
            return

        # Get column names from cursor description
        column_names = [description[0] for description in cursor.description]

        print(f"--- Last {len(rows)} '{kind}' events ---")
        for row in rows:
            event_data = dict(zip(column_names, row))
            print(event_data)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # The PMM object handles its own DB connection lifecycle, so we don't close it here.
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the PMM event log.")
    parser.add_argument(
        "kind",
        type=str,
        help="The kind of event to query (e.g., 'commitment', 'evidence', 'all').",
    )
    parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=5,
        help="The maximum number of events to return.",
    )
    args = parser.parse_args()

    query_events(args.kind, args.limit)
