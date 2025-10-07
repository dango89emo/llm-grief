#!/usr/bin/env python3
"""
View all diaries in chronological order.
"""

import sys
from pathlib import Path


def view_diaries(persona_id: int = 1, base_dir: str = "data") -> None:
    """
    Display all diaries for a persona in chronological order.

    Args:
        persona_id: Persona ID to view
        base_dir: Base directory containing diary data
    """
    persona_dir = Path(base_dir) / f"persona_{persona_id}"

    if not persona_dir.exists():
        print(f"Error: Persona directory not found: {persona_dir}")
        sys.exit(1)

    baseline_dir = persona_dir / "baseline"
    grief_dir = persona_dir / "grief"

    # Collect all diary files
    all_diaries = []

    if baseline_dir.exists():
        all_diaries.extend(sorted(baseline_dir.glob("day*.txt")))

    if grief_dir.exists():
        all_diaries.extend(sorted(grief_dir.glob("day*.txt")))

    if not all_diaries:
        print(f"No diaries found for persona {persona_id}")
        sys.exit(1)

    # Sort by filename to ensure chronological order
    all_diaries.sort(key=lambda x: x.name)

    # Display header
    print("=" * 80)
    print(f"Diaries for Persona {persona_id}")
    print("=" * 80)

    # Display loss event if exists
    loss_event_path = persona_dir / "loss_event.txt"
    if loss_event_path.exists():
        loss_event = loss_event_path.read_text(encoding="utf-8").strip()
        print(f"\n【喪失イベント】\n{loss_event}\n")

    # Display each diary
    for diary_path in all_diaries:
        # Extract day number from filename (e.g., day0001.txt -> 1)
        day_num = int(diary_path.stem.replace("day", ""))

        # Determine diary type
        diary_type = "baseline" if diary_path.parent.name == "baseline" else "grief"

        # Read diary content
        content = diary_path.read_text(encoding="utf-8").strip()

        # Display diary
        print("-" * 80)
        print(f"Day {day_num:04d} [{diary_type}]")
        print("-" * 80)
        print(content)
        print()

    print("=" * 80)
    print(f"Total: {len(all_diaries)} diaries")
    print("=" * 80)


def main():
    """Main function."""
    persona_id = 1

    # Check if persona_id provided as argument
    if len(sys.argv) > 1:
        try:
            persona_id = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid persona ID: {sys.argv[1]}")
            print("Usage: python view_diaries.py [persona_id]")
            sys.exit(1)

    view_diaries(persona_id)


if __name__ == "__main__":
    main()
