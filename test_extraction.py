"""Test the improved PDF extraction against curriculum files."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.core.ingestion import load_pdf, extract_program_info, find_section_headers_for_tables

# Test 1: Program info extraction
print("=== Program Info Extraction ===")
tests = [
    "CURRICULUM FOR BACHELOR OF SCIENCE IN ARCHITECTURE (BS ARCH).pdf",
    "CURRICULUM FOR BACHELOR OF SCIENCE IN MATHEMATICS (BS MATH).pdf",
    "CURRICULUM FOR BACHELOR OF SCIENCE IN ELECTRONICS ENGINEERING (BS ECE).pdf",
    "2025 ADNU COLLEGE HANDBOOK - CHAPTER 4.pdf",
]
for fn in tests:
    info = extract_program_info(fn)
    print(f"  {fn[:50]:50s} -> {info}")

# Test 2: Full extraction on Architecture curriculum
print("\n=== Architecture Curriculum Extraction ===")
path = r"documents\CURRICULUM FOR BACHELOR OF SCIENCE IN ARCHITECTURE (BS ARCH).pdf"
fn = "CURRICULUM FOR BACHELOR OF SCIENCE IN ARCHITECTURE (BS ARCH).pdf"
docs = load_pdf(path, fn)

print(f"Total chunks: {len(docs)}")
tables = [d for d in docs if d.metadata.get("type") == "table"]
texts = [d for d in docs if d.metadata.get("type") == "text"]
print(f"  Table chunks: {len(tables)}")
print(f"  Text chunks:  {len(texts)}")

for i, d in enumerate(docs[:5]):
    print(f"\n--- Chunk {i+1} ({d.metadata['type']}, page {d.metadata['page']}) ---")
    print(f"  Metadata: {d.metadata}")
    preview = d.page_content[:300].replace('\n', ' | ')
    print(f"  Content:  {preview}...")

# Test 3: Check that body text does NOT contain table content
print("\n=== Body Text Duplication Check ===")
for d in texts:
    # Check for signs of garbled table text (course codes mashed together)
    if "CourseCode" in d.page_content.replace(" ", "") and len(d.page_content) > 200:
        print(f"  WARNING: Page {d.metadata['page']} body text may contain table content!")
        print(f"  Preview: {d.page_content[:150]}")
    else:
        print(f"  Page {d.metadata['page']}: Clean ({len(d.page_content)} chars)")

# Test 4: Math curriculum (was previously most garbled)
print("\n=== Math Curriculum Extraction ===")
path2 = r"documents\CURRICULUM FOR BACHELOR OF SCIENCE IN MATHEMATICS (BS MATH).pdf"
fn2 = "CURRICULUM FOR BACHELOR OF SCIENCE IN MATHEMATICS (BS MATH).pdf"
docs2 = load_pdf(path2, fn2)
tables2 = [d for d in docs2 if d.metadata.get("type") == "table"]
texts2 = [d for d in docs2 if d.metadata.get("type") == "text"]
print(f"Total: {len(docs2)} chunks ({len(tables2)} tables, {len(texts2)} text)")

for d in docs2[:3]:
    print(f"\n--- {d.metadata['type']} (page {d.metadata['page']}) ---")
    preview = d.page_content[:250].replace('\n', ' | ')
    print(f"  {preview}")

print("\nDone.")
