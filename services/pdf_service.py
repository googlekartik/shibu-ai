"""PDF generation with ReportLab. Files go to generated/ only."""
import os
import time
import uuid
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT

from utils.security import sanitize_filename


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(15 * mm, 12 * mm, "Rule-Based Intelligent Chatbot")
    canvas.drawRightString(A4[0] - 15 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


class PDFService:
    def __init__(self, output_dir=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = output_dir or os.path.join(base, "generated")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pdf(self, content: str, title: str = "Chatbot Export"):
        """Return (absolute_path, filename)."""
        safe = sanitize_filename(title)
        filename = f"{safe}_{int(time.time())}_{uuid.uuid4().hex[:6]}.pdf"
        path = os.path.join(self.output_dir, filename)

        styles = {
            "title": ParagraphStyle("Title2", fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=6),
            "meta": ParagraphStyle("Meta", fontName="Helvetica", fontSize=9, leading=12, spaceAfter=10),
            "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=11, leading=15, alignment=TA_LEFT, spaceAfter=6),
        }
        story = [
            Paragraph(title, styles["title"]),
            Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["meta"]),
            Spacer(1, 4 * mm),
        ]
        for para in (content or "").split("\n"):
            para = para.strip()
            if not para:
                continue
            # Escape for ReportLab paragraph markup
            para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(para, styles["body"]))
        doc = SimpleDocTemplate(path, pagesize=A4,
                                leftMargin=18 * mm, rightMargin=18 * mm,
                                topMargin=18 * mm, bottomMargin=18 * mm,
                                title=title)
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return path, filename


def generate_pdf(content, title="Chatbot Export"):
    """Module-level helper matching the brief's generate_pdf(content, title)."""
    return PDFService().generate_pdf(content, title)
