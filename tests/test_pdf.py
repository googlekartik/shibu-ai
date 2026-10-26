import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.pdf_service import PDFService


def test_pdf_generated(tmp_path):
    svc = PDFService(output_dir=str(tmp_path))
    path, filename = svc.generate_pdf("Hello PDF test.", title="Test")
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    assert filename.endswith(".pdf")
