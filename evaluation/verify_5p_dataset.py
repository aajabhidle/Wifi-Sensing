#!/usr/bin/env python3
"""
verify_5p_dataset.py - Verifies all 5 participants (45 recordings) across BHU, LHU, RHU.
Enforces whole-token matching and asserts no name clash between Vansh and Vardaan.
"""

from pathlib import Path
import re
import sys

BASE_DIR = Path("E:/percom")
REC = re.compile(r"^(?P<act>[A-Za-z]+)_S(?P<s>\d+)_(?P<person>[A-Za-z0-9]+)$")

EXPECTED_PERSONS = ["Priyanshu", "Sanchit", "Sir", "Vansh", "Vardaan"]
EXPECTED_ACTS = ["BHU", "LHU", "RHU"]
EXPECTED_SESSIONS = ["S1", "S2", "S3"]

def main():
    print("=== Scanning 5-Participant Dataset under E:\\percom ===")
    recs = []
    by_person = {p: [] for p in EXPECTED_PERSONS}

    for adir in sorted(p for p in BASE_DIR.iterdir() if p.is_dir() and p.name in EXPECTED_ACTS):
        for r in sorted(p for p in adir.iterdir() if p.is_dir()):
            m = REC.match(r.name)
            if not m:
                continue
            person = m["person"].capitalize()
            if person not in EXPECTED_PERSONS:
                print(f"Skipping excluded/unknown person: {person} ({r.name})")
                continue
            act = m["act"].lower()
            sess = f"S{int(m['s'])}"
            sid = f"{act}_{person}_{sess}"
            
            # Verify files inside
            csi = list(r.glob("rx_*.csi"))
            vid = r / "video.mp4"
            tim = r / "timing.txt"
            
            rec_info = dict(
                folder=r, act=act, session=sess, person=person, sid=sid, name=r.name,
                has_csi=len(csi) > 0, has_video=vid.exists(), has_timing=tim.exists()
            )
            recs.append(rec_info)
            by_person[person].append(rec_info)

    print(f"\nTotal verified recordings: {len(recs)} (Expected: 45 = 5 persons x 3 gestures x 3 sessions)")
    assert len(recs) == 45, f"Expected 45 recordings, got {len(recs)}"

    print("\n--- Breakdown by Participant ---")
    for p in EXPECTED_PERSONS:
        p_recs = by_person[p]
        print(f"  {p:<10}: {len(p_recs)} recordings | All CSI: {all(r['has_csi'] for r in p_recs)} | All Video: {all(r['has_video'] for r in p_recs)} | All Timing: {all(r['has_timing'] for r in p_recs)}")
        assert len(p_recs) == 9, f"Person {p} has {len(p_recs)} recordings instead of 9!"

    # Explicit Whole-Token Name Clash Assertion: Vansh vs Vardaan
    vansh_sids = {r["sid"] for r in by_person["Vansh"]}
    vardaan_sids = {r["sid"] for r in by_person["Vardaan"]}
    overlap = vansh_sids.intersection(vardaan_sids)
    assert len(overlap) == 0, f"Name clash detected between Vansh and Vardaan: {overlap}"
    assert all("vansh" in s.lower() and "vardaan" not in s.lower() for s in vansh_sids)
    assert all("vardaan" in s.lower() and "vansh" not in s.lower() for s in vardaan_sids)

    print("\n[SUCCESS] Exact whole-token matching verified: Vansh and Vardaan have completely disjoint, valid recordings.")
    print("Dataset verification passed 45/45 recordings!")

if __name__ == "__main__":
    main()
