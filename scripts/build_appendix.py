"""Build the model-neutral supplemental methods PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "appendix.pdf"


def page_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B5BDC4"))
    canvas.line(48, 48, 564, 48)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#56616B"))
    canvas.drawString(48, 34, "HypoRAG | Supplemental methods")
    canvas.drawRightString(564, 34, str(document.page))
    canvas.restoreState()


def build() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=13,
        textColor=colors.HexColor("#193A44"),
    ))
    styles.add(ParagraphStyle(
        name="SectionCustom", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=11.5, leading=15, spaceBefore=16, spaceAfter=7,
        textColor=colors.HexColor("#193A44"),
    ))
    styles.add(ParagraphStyle(
        name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=13.5, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="NoteCustom", parent=styles["BodyCustom"], fontSize=8.5,
        leading=12, textColor=colors.HexColor("#48545B"),
    ))
    body = styles["BodyCustom"]
    note = styles["NoteCustom"]
    section = styles["SectionCustom"]

    story = [
        Paragraph("HypoRAG: Supplemental Methods and Reproduction Notes", styles["TitleCustom"]),
        Paragraph(
            "This document records the definitions and implementation details behind "
            "the top-1 retrieval comparison and the released pipeline code. "
            "The paired benchmark data and generated knowledge base are distributed separately.",
            body,
        ),
        Paragraph("A. Top-1 retrieval study", section),
        Paragraph(
            "Fifty vulnerable functions were sampled from the PrimeVul-v0.1 test "
            "split with a fixed seed. The candidate corpus contained 3,611 "
            "eligible vulnerable functions from the training split. Random "
            "selection, BM25 over complete "
            "function code, dense code retrieval, and dense retrieval over "
            "generated functional summaries each returned one historical case "
            "per function. Hypothesis-guided retrieval returned one case for "
            "each of 141 hypotheses from the same 50 functions. Cases sharing "
            "a project or CVE with the query, or with token Jaccard at least "
            "0.80, were excluded. All 341 pairs are in retrieval_diagnostic/.",
            body,
        ),
    ]
    table_data = [
        ["Retriever", "N", "Token Sim.", "Call Overlap", "Mech. Match"],
        ["Random", "50", "0.0436", "0.0000", "4/50 (8%)"],
        ["BM25", "50", "0.0899", "0.0494", "10/50 (20%)"],
        ["Dense-Code", "50", "0.0887", "0.0516", "6/50 (12%)"],
        ["Dense-LLM", "50", "0.0846", "0.0399", "10/50 (20%)"],
        ["Hypothesis-Guided", "141", "0.0621", "0.0235", "114/141 (80.9%)"],
    ]
    table = Table(table_data, colWidths=[149, 35, 98, 108, 124], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE7E7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#193A44")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B5BDC4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        table,
        Spacer(1, 8),
        Paragraph(
            "Table A1. Filtered top-1 retrieval results. The function-level "
            "methods share 50 targets; Hypothesis-Guided uses 141 hypotheses "
            "from those targets. The supplied manuscript prints 0.0400 for "
            "Random Token Sim.; the released rows recompute to 0.0436.",
            note,
        ),
        Paragraph("B. Surface similarity", section),
        Paragraph(
            "Token similarity is the archived set Jaccard score on lexical "
            "tokens: |Tq intersection Tc| / |Tq union Tc|. Call overlap is "
            "asymmetric: |Cq intersection Cc| / |Cq|, using distinct query-side "
            "and case-side callee names. The released per-pair scores allow "
            "independent recomputation of the means in Table A1.",
            body,
        ),
        Paragraph("C. Mechanism judgment", section),
        Paragraph(
            "The audit has three levels: 0 for unrelated, 1 for a transferable "
            "verification principle, and 2 for the same decisive failed "
            "condition. Table A1 counts levels 1 and 2 as mechanism matches. "
            "Shared CWE alone does not establish a match. The audit packets "
            "include descriptions, repair diffs, and a rationale per pair.",
            body,
        ),
        Paragraph(
            "The released file records the final human-adjudicated "
            "three-level label and rationale for each of the 200 "
            "function-level and 141 hypothesis-level pairs. Two annotators "
            "independently applied the protocol, and a third adjudicated "
            "disagreements.",
            note,
        ),
        Paragraph("D. Structured knowledge and online inference", section),
        Paragraph(
            "For each training repair pair, the pipeline extracts a repair "
            "signature, then independently constructs retrieval fields and "
            "verification guidance. The retrieval fields describe mechanism, "
            "repair, and decisive evidence. They are indexed as separate dense "
            "views. The guidance fields are case explanation, vulnerable pattern, "
            "repaired pattern, and verification guidance.",
            body,
        ),
        Paragraph(
            "At test time, a single vulnerable or safe function is processed "
            "without its paired counterpart. The detector proposes localized "
            "hypotheses, retrieves candidates from three views within an "
            "assigned family, deduplicates by knowledge ID, reranks the union, "
            "and passes the selected guidance fields to point verification. "
            "An option in the formal runner aggregates supported point "
            "judgments using a deterministic Boolean OR.",
            body,
        ),
        Paragraph("E. Reranker and evaluation bookkeeping", section),
        Paragraph(
            "Teacher utility labels take values 0, 1, or 2. Canonical "
            "pairwise training examples are formed only within complete "
            "query groups and only between candidates with unequal utility "
            "labels. The training objective weights pairwise logistic loss by "
            "the label gap. Ranking evaluation can report micro preference "
            "accuracy, macro per-query preference accuracy, MRR, NDCG, and "
            "recall on the same test groups before and after fine-tuning.",
            body,
        ),
        Paragraph(
            "Pair outcomes are exhaustive for valid vulnerability/repair "
            "pairs: Both-R (vulnerable side flagged, repaired side safe), "
            "Both-W (reverse), Both-s (both safe), and Both-v (both flagged). "
            "The four counts sum to the number of valid pairs. Incomplete "
            "or unparseable records are reported separately and are not "
            "silently placed into any outcome class.",
            body,
        ),
        Paragraph("F. Reproducibility boundary", section),
        Paragraph(
            "The release includes all 341 query/case pairs, final audit "
            "labels, rationales, and metric code. A reader can recompute "
            "Table A1 offline. Re-running dense retrieval or generation from "
            "scratch requires the original benchmark, chosen local encoder, "
            "and configured inference endpoint. The release intentionally "
            "omits private credentials, model weights, and the generated "
            "knowledge base.",
            body,
        ),
        Paragraph("G. Artifact field guide", section),
        Paragraph(
            "retrieval_pairs.jsonl stores method, query and candidate IDs, "
            "ranking score, surface scores, exclusion flags, and both "
            "vulnerable function bodies. mechanism_audit.jsonl adds the "
            "hypothesis where applicable, descriptions, repair diffs, "
            "three-level label, and rationale for all 341 pairs.",
            body,
        ),
        Paragraph(
            "The semantic summary file contains sanitized functional "
            "summaries for the training candidates and selected test queries. "
            "It omits model identifier, raw response, and billing fields. "
            "The manifest fixes the 50 query IDs and exclusion settings; "
            "candidate_ids.json fixes the 3,611-item candidate corpus.",
            body,
        ),
        Paragraph("H. Failure and coverage reporting", section),
        Paragraph(
            "The formal runner writes append-only stage outputs with "
            "input hashes and success or failure status. Resumption skips "
            "successful records and retries only selected failures. "
            "A valid end-to-end pair requires successful processing of "
            "both function sides; reported confusion counts and paired "
            "outcomes use the same valid-pair denominator. Excluded inputs "
            "and parsing failures remain explicit in the stage reports.",
            body,
        ),
    ])

    document = BaseDocTemplate(
        str(OUTPUT), pagesize=(612, 792), leftMargin=48, rightMargin=48,
        topMargin=48, bottomMargin=60, title="HypoRAG supplemental methods",
        author="HypoRAG", subject="Methods and reproducibility details",
    )
    frame = Frame(48, 60, 516, 684, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
    document.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=page_footer))
    document.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
