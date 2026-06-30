import re


def build_user_prompt(data: dict) -> str:
    """Собирает ответы пользователя в исходные данные для разбора (блоки 1–5)."""
    template = (
        "Вот исходные данные для анализа проекта (выполни блоки 1–5):\n"
        "Название проекта: {project_name}\n"
        "Идея проекта: {idea}\n"
        "Территория реализации: {territory}\n"
        "Целевая аудитория: {audience}\n"
        "Возраст/особенности аудитории: {age_features}\n"
        "Какая проблема предполагается: {problem}\n"
        "Какие данные уже есть (подтверждение проблемы): {existing_data}\n"
        "Каких данных пока не хватает: {missing_data}\n"
        "Срок проекта: {duration}\n\n"
        "Команда, партнёры и бюджет не указаны — трактуй их как гипотезу "
        "по правилам системного промпта. Если срок указан как «не знаю» — "
        "прими гипотезу 6–8 месяцев и пометь её.\n\n"
        "Выполни РОВНО блоки 1–5 строго по порядку, за один ответ. "
        "Не останавливайся между блоками, не задавай вопросов в конце."
    )
    return template.format(
        project_name=data.get("project_name", "не указано"),
        idea=data.get("idea", "не указано"),
        territory=data.get("territory", "не указано"),
        audience=data.get("audience", "не указано"),
        age_features=data.get("age_features", "не указано"),
        problem=data.get("problem", "не указано"),
        existing_data=data.get("existing_data", "нет данных"),
        missing_data=data.get("missing_data", "неизвестно"),
        duration=data.get("duration", "не указано"),
    )


# ──────────────────────────────────────────────────────────────────────────
# Форматирование ответа модели под мессенджер MAX
# ──────────────────────────────────────────────────────────────────────────

_HR_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_QUOTE_RE = re.compile(r"^\s*>\s?")
_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}")


def _strip_inline_markup(s: str) -> str:
    s = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"_(.+?)_", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", s)
    return s


def format_for_max(text: str, markdown: bool = True) -> str:
    out_lines = []
    in_code = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue

        if in_code:
            out_lines.append(line)
            continue

        if _HR_RE.match(line):
            continue
        if "|" in line and _TABLE_SEP_RE.match(line):
            continue
        line = _QUOTE_RE.sub("", line)

        m = _HEADER_RE.match(line)
        if m:
            title = _strip_inline_markup(m.group(2).strip())
            out_lines.append(f"**{title}**" if markdown else title)
            continue

        line = _BULLET_RE.sub(r"\1• ", line)
        out_lines.append(line)

    text = "\n".join(out_lines)

    if markdown:
        text = re.sub(r"\*\*\*(.+?)\*\*\*", r"**\1**", text)
        text = re.sub(r"(?<!_)__(?!_)(.+?)(?<!_)__(?!_)", r"**\1**", text)
    else:
        text = _strip_inline_markup(text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()