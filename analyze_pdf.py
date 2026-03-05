"""Quick analysis of curriculum PDF extraction quality."""
import pdfplumber, fitz

pdfs = [
    r'documents\CURRICULUM FOR BACHELOR OF SCIENCE IN ARCHITECTURE (BS ARCH).pdf',
    r'documents\CURRICULUM FOR BACHELOR OF SCIENCE IN MATHEMATICS (BS MATH).pdf',
    r'documents\CURRICULUM FOR BACHELOR OF SCIENCE IN ELECTRONICS ENGINEERING (BS ECE).pdf',
]

for path in pdfs:
    print(f"\n{'='*60}")
    print(f"FILE: {path.split(chr(92))[-1]}")
    print('='*60)
    
    doc = fitz.open(path)
    print(f"Pages: {len(doc)}")
    
    for i in range(min(3, len(doc))):
        text = doc[i].get_text("text")
        print(f"\n--- Page {i+1} (PyMuPDF: {len(text)} chars) ---")
        preview = text[:200].replace('\n', ' | ') if text else '[EMPTY]'
        print(f"  Preview: {preview}")
    doc.close()
    
    pdf = pdfplumber.open(path)
    for i, page in enumerate(pdf.pages[:3]):
        tables = page.find_tables()
        words = page.extract_words()
        print(f"\n  Page {i+1}: {len(tables)} tables, {len(words)} words detected by pdfplumber")
        for j, t in enumerate(tables):
            data = t.extract()
            if data:
                print(f"    Table {j+1}: {len(data)} rows x {len(data[0])} cols")
                print(f"    Header: {data[0][:4]}...")
    pdf.close()

print("\nDone.")
