import os
import re
import logging
from datetime import datetime
from fpdf import FPDF

logger = logging.getLogger(__name__)

_FONT_DIR = os.path.dirname(os.path.abspath(__file__))
_FONT_REGULAR = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
_FONT_BOLD = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")

_HEADING_MD_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HEADING_BOLD_RE = re.compile(r"^\*\*(.+)\*\*$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+)$")
_HR_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_QUOTE_RE = re.compile(r"^\s*>+\s?")
_BLOCK_NUM_RE = re.compile(r"^БЛОК\s+\d+", re.IGNORECASE)


def _strip_inline(s: str) -> str:
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"_(.+?)_", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", s)
    return s.strip()


def _block_title(line: str):
    s = _strip_inline(line.strip())
    s = s.lstrip("#>*-• \t").strip()
    if _BLOCK_NUM_RE.match(s):
        return s
    return None


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") or s.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    s = line.strip().strip("|").replace(" ", "")
    return bool(s) and set(s) <= {"-", ":"}


def _parse_table_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [_strip_inline(c.strip()) for c in s.split("|")]


def _ensure_page(pdf):
    """Гарантирует, что страница открыта, иначе вызывает add_page()."""
    if pdf.page_no() == 0:
        pdf.add_page()


def _safe_multi_cell(pdf, w, h, text, markdown=False, align="L"):
    _ensure_page(pdf)
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(w, h, text, markdown=markdown, align=align)
        return
    except Exception:
        pass
    # fallback
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(w, h, text, markdown=markdown, wrapmode="CHAR", align=align)
        return
    except Exception:
        pass
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(w, h, _strip_inline(text), wrapmode="CHAR", align=align)
    except Exception:
        pass


def _render_table(pdf, block: list[str]):
    _ensure_page(pdf)
    rows = [_parse_table_cells(ln) for ln in block if not _is_separator_row(ln)]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    font_size = 8 if ncols <= 5 else 7 if ncols <= 8 else 6
    try:
        pdf.set_font("DejaVu", "", font_size)
        with pdf.table(
            first_row_as_headings=True,
            markdown=True,
            wrapmode="CHAR",
            padding=1,
        ) as table:
            for r in rows:
                trow = table.row()
                for cell in r:
                    trow.cell(cell)
        pdf.ln(2)
    except Exception:
        pdf.set_font("DejaVu", "", 9)
        for r in rows:
            _safe_multi_cell(pdf, 0, 5, " | ".join(r))
        pdf.ln(2)
    finally:
        pdf.set_font("DejaVu", "", 12)


def _render_lines(pdf, lines, a, b):
    """Рендерит строки lines[a..b] (включительно) с автоматическим добавлением страницы."""
    _ensure_page(pdf)
    n = len(lines)
    i = a
    while i <= b:
        raw = lines[i].rstrip()
        stripped = _QUOTE_RE.sub("", raw).strip()

        if not stripped:
            pdf.ln(3)
            i += 1
            continue

        # Заголовок блока
        bt = _block_title(raw)
        if bt:
            pdf.ln(2)
            pdf.set_font("DejaVu", "B", 14)
            _safe_multi_cell(pdf, 0, 7, bt)
            pdf.ln(1)
            pdf.set_font("DejaVu", "", 12)
            i += 1
            continue

        # Таблица
        if _is_table_row(raw) and i + 1 < n and _is_table_row(lines[i + 1]):
            block_lines = []
            while i <= b and _is_table_row(lines[i]):
                block_lines.append(lines[i])
                i += 1
            _render_table(pdf, block_lines)
            continue

        # Горизонтальный разделитель
        if _HR_RE.match(stripped):
            i += 1
            continue

        # Под-заголовок
        m = _HEADING_MD_RE.match(stripped)
        bold_head = _HEADING_BOLD_RE.match(stripped) if not m else None
        if m or bold_head:
            title_text = _strip_inline(m.group(2) if m else bold_head.group(1))
            level = len(m.group(1)) if m else 3
            pdf.set_font("DejaVu", "B", 13 if level <= 2 else 12)
            pdf.ln(2)
            _safe_multi_cell(pdf, 0, 6, title_text)
            pdf.ln(1)
            pdf.set_font("DejaVu", "", 12)
            i += 1
            continue

        # Маркированный список
        bl = _BULLET_RE.match(raw)
        if bl:
            _ensure_page(pdf)
            pdf.set_x(pdf.l_margin + 6)
            _safe_multi_cell(pdf, 0, 6, "•   " + bl.group(1), markdown=True)
            pdf.set_x(pdf.l_margin)
            i += 1
            continue

        # Обычный абзац
        _ensure_page(pdf)
        pdf.set_font("DejaVu", "", 12)
        _safe_multi_cell(pdf, 0, 6, stripped, markdown=True)
        i += 1


class PDFWithFooter(FPDF):
    def __init__(self, project_name, date_str):
        super().__init__()
        self.project_name = project_name
        self.date_str = date_str
        self.set_auto_page_break(auto=True, margin=15)
        # Обычный шрифт обязателен: без него кириллица не отрендерится.
        if not os.path.exists(_FONT_REGULAR):
            raise FileNotFoundError(
                f"Не найден шрифт {_FONT_REGULAR}. Положите DejaVuSans.ttf "
                "рядом с pdf_generator.py (или поправьте _FONT_REGULAR)."
            )
        self.add_font("DejaVu", "", _FONT_REGULAR)

        # Жирный — опционален.
        if os.path.exists(_FONT_BOLD):
            self.add_font("DejaVu", "B", _FONT_BOLD)
        else:
            logger.warning(
                "Не найден %s — жирное начертание будет заменено обычным.",
                _FONT_BOLD,
            )
            self.add_font("DejaVu", "B", _FONT_REGULAR)

    def footer(self):
        _ensure_page(self)
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(140, 140, 140)
        self.cell(
            0, 10, f"{self.project_name} • {self.date_str} • стр. {self.page_no()}",
            align="C"
        )
        self.set_text_color(0, 0, 0)


def _title_page(pdf, project_name, subtitle, date_str, intro):
    pdf.add_page()
    pdf.ln(45)
    pdf.set_font("DejaVu", "B", 22)
    pdf.multi_cell(0, 12, project_name, align="C")
    pdf.ln(4)
    pdf.set_draw_color(180, 180, 180)
    y = pdf.get_y()
    pdf.line(pdf.w / 2 - 45, y, pdf.w / 2 + 45, y)
    pdf.ln(8)

    pdf.set_font("DejaVu", "", 15)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 9, subtitle, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    pdf.set_font("DejaVu", "", 12)
    pdf.multi_cell(0, 8, f"Дата: {date_str}", align="C")
    pdf.ln(16)

    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 7, intro, align="C")
    pdf.set_text_color(0, 0, 0)


def generate_pdf_from_text(
    text: str,
    project_name: str = "Экспертный анализ",
    date_str: str = None,
    free_blocks: int = None,
) -> bytes:
    if date_str is None:
        date_str = datetime.now().strftime("%d.%m.%Y")
    lines = text.split("\n")

    # ---- Распознаём блоки ----
    blocks = []
    current = None
    bstart = 0
    for i, line in enumerate(lines):
        t = _block_title(line)
        if t:
            if current is not None:
                blocks.append((current, bstart, i - 1))
            current = t
            bstart = i
    if current is not None:
        blocks.append((current, bstart, len(lines) - 1))

    pdf = PDFWithFooter(project_name, date_str)

    subtitle = "Экспертный анализ социального проекта"
    if free_blocks:
        subtitle += " (сокращённая версия)"
    intro = (
        f"Это {'сокращённый бесплатный разбор — блоки 1–' + str(free_blocks) + ' из 20' if free_blocks else 'полный разбор по 20 блокам'}. "
        "Документ содержит экспертные выводы и рекомендации."
    )

    _title_page(pdf, project_name, subtitle, date_str, intro)

    if blocks:
        # ---- Оглавление ----
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 16)
        pdf.multi_cell(0, 10, "Содержание", align="L")
        pdf.ln(3)
        pdf.set_font("DejaVu", "", 12)
        for title, _, _ in blocks:
            short = title[:70] + "…" if len(title) > 70 else title
            _safe_multi_cell(pdf, 0, 8, "•   " + short)
        pdf.ln(3)

        # Преамбула
        first_start = blocks[0][1]
        if first_start > 0:
            pdf.add_page()
            _render_lines(pdf, lines, 0, first_start - 1)

        # ---- Контент по блокам ----
        for title, start, end in blocks:
            pdf.add_page()
            pdf.ln(2)
            pdf.set_font("DejaVu", "B", 15)
            _safe_multi_cell(pdf, 0, 8, title)
            pdf.ln(2)
            pdf.set_font("DejaVu", "", 12)
            _render_lines(pdf, lines, start + 1, end)
            pdf.ln(4)
    else:
        # ---- Блоки не распознаны — выводим весь текст ----
        pdf.add_page()
        _render_lines(pdf, lines, 0, len(lines) - 1)

    return bytes(pdf.output())