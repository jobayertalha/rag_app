"""
ingest.py — Run once to build the FAISS index from jd_knowledge_base.json.

Each role's jd_text field is embedded so FAISS matches CVs against
real job description language — not hand-written summaries.

Usage:
    python ingest.py

This downloads ~80MB embedding model (all-MiniLM-L6-v2) on first run.
Subsequent runs are fast.
"""

import os


def ensure_index():
    """Auto-build index on first run (called by app.py if needed)."""
    if not os.path.exists("faiss_index"):
        print("Building FAISS index from JD texts...")
        from rag import load_roles, build_index
        roles = load_roles()
        build_index(roles)
        print(f"Done. Indexed {len(roles)} job descriptions.")


if __name__ == "__main__":
    from rag import load_roles, build_index

    print("=" * 50)
    print("AI Career Match — FAISS Index Builder")
    print("=" * 50)
    print()
    print("Loading jd_knowledge_base.json...")
    roles = load_roles("jd_knowledge_base.json")
    print(f"Found {len(roles)} roles with JD texts.")
    print()
    print("Building FAISS index...")
    print("(Downloads ~80MB embedding model on first run)")
    print()
    build_index(roles)
    print()
    print("=" * 50)
    print(f"Done! Index saved to ./faiss_index/")
    print("Each role's jd_text was embedded.")
    print("FAISS now matches CVs against real JD language.")
    print()
    print("To add your own JDs:")
    print("  1. Open jd_knowledge_base.json")
    print("  2. Replace any role's jd_text with a real JD")
    print("  3. Run: python ingest.py")
    print()
    print("Next: streamlit run app.py")
    print("=" * 50)
