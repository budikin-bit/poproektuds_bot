import logging
import unicodedata
from config import MASK_PII
from pii import mask_pii
from session_manager import get_session, mutate_session

logger = logging.getLogger(__name__)

# Только вопросы, нужные для блоков 1–5. Срок добавлен по просьбе.
QUESTIONS = [
    {
        "id": "project_name",
        "title": "Рабочее название проекта",
        "text": (
            "Дайте рабочее название проекта (кратко, ёмко).\n"
            "Оно будет на титульной странице анализа.\n"
            "Пример: «Робототехника для сельских школьников»."
        ),
    },
    {
        "id": "idea",
        "title": "Идея проекта",
        "text": (
            "Кратко опишите идею: что и для кого вы хотите сделать.\n"
            "Например: «Занятия цифровым рисунком для подростков в сельской школе»."
        ),
    },
    {
        "id": "territory",
        "title": "Территория",
        "text": (
            "Где будет реализован проект — село/город, район, регион, площадки?\n"
            "Например: «с. Ытык-Кюель, Таттинский улус, Республика Саха (Якутия)»."
        ),
    },
    {
        "id": "audience",
        "title": "Аудитория",
        "text": (
            "Кто именно получит пользу? Чем конкретнее, тем лучше.\n"
            "Например: «Подростки 12–17 лет, школа №1». СТРОГО БЕЗ ПЕРСОНАЛЬНЫХ ДАННЫХ."
        ),
    },
    {
        "id": "age_features",
        "title": "Возраст и особенности",
        "text": (
            "Возраст и особенности аудитории: ОВЗ, трудная ситуация, миграция и т.п.\n"
            "Если особенностей нет — «нет».\n"
            "Например: «12–17 лет, есть слабослышащие». СТРОГО БЕЗ ПЕРСОНАЛЬНЫХ ДАННЫХ."
        ),
    },
    {
        "id": "problem",
        "title": "Проблема",
        "text": (
            "Какую проблему решает проект и в чём она проявляется?\n"
            "Например: «Подросткам негде применять цифровые навыки — нет среды и оборудования»."
        ),
    },
    {
        "id": "existing_data",
        "title": "Подтверждение проблемы (данные/опрос)",
        "text": (
            "Чем подтверждается проблема: опрос, статистика, обращения, ваш опыт?\n"
            "⚠️ Если приводите проценты — укажите, сколько человек ответило (по каждой группе отдельно): "
            "без размера выборки точные числа в разбор не попадут.\n"
            "Например: «Опрос N=356: 37,1% (132 чел.) хотят учиться цифровому рисунку»."
        ),
    },
    {
        "id": "missing_data",
        "title": "Каких данных не хватает",
        "text": (
            "Каких данных или цифр не хватает, чтобы обосновать проблему?\n"
            "Если не знаете — «не знаю».\n"
            "Например: «Нет данных о реальной готовности ходить регулярно»."
        ),
    },
    {
        "id": "duration",
        "title": "Срок проекта",
        "text": (
            "Срок проекта — даты или длительность. Одной строкой.\n"
            "Если ещё не решили — «не знаю».\n"
            "Например: «9 месяцев, март–ноябрь 2026»."
        ),
    },
]

TOTAL_STEPS = len(QUESTIONS)

CHAR_LIMITS = {
    "project_name": 200,
    "idea": 1500,
    "territory": 300,
    "audience": 1000,
    "age_features": 1500,
    "problem": 2000,
    "existing_data": 7000,
    "missing_data": 1000,
    "duration": 350,
}

MIN_NEWLINES = 15
_CHARS_PER_NEWLINE = 40


def _max_newlines_for(q_id) -> int:
    limit = CHAR_LIMITS.get(q_id)
    return MIN_NEWLINES if not limit else max(MIN_NEWLINES, limit // _CHARS_PER_NEWLINE)


def normalize_answer(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch >= " " or ch in "\n\t")
    return text.strip()


def _validate_answer(answer: str, q_id=None) -> tuple[bool, str]:
    max_newlines = _max_newlines_for(q_id)
    nl = answer.count("\n")
    if nl > max_newlines:
        logger.warning("Слишком много переносов: %s (лимит %s, поле %s)", nl, max_newlines, q_id)
        return False, (
            f"✂️ Слишком много переносов строк ({nl}). Уберите лишние пустые строки."
        )
    return True, ""


def get_question_text(step: int, total: int = TOTAL_STEPS) -> str:
    q = QUESTIONS[step]
    return f"📋 Шаг {step + 1}/{total}. {q['title']}.\n{q['text']}"


def get_question_id(step: int) -> str:
    return QUESTIONS[step]["id"]


def get_keyboard_for_step(step: int):
    """Все шаги — свободный ввод; клавиатуры не используются."""
    return None


def process_answer(user_id: str, answer: str):
    """
    Обработать ответ. (None, None) — переходим к анализу.
    ВАЖНО: тяжёлая маскировка ПД (natasha-NER в mask_pii) выполняется ВНЕ
    db.lock — иначе на время NER блокируется единый замок БД для всех чатов.
    Корректность read-modify-write обеспечивается:
      • per-chat сериализацией в bot._chat_lock(chat_id) (один ответ за раз),
      • повторной проверкой шага уже под db.lock внутри _apply.
    Саму функцию в bot.py стоит вызывать через asyncio.to_thread, чтобы NER и
    запись в sqlite не блокировали event loop.
    """
    session = get_session(user_id)
    step = session.get("step", 0)
    if session.get("finished") or step >= TOTAL_STEPS:
        return None, None

    q_id = get_question_id(step)
    clean = normalize_answer(answer)

    if not clean:
        return (
            "✍️ Кажется, ответ пустой. Напишите, пожалуйста, текстом.\n\n"
            + get_question_text(step),
            None,
        )

    ok, err = _validate_answer(clean, q_id)
    if not ok:
        return err + "\n\n" + get_question_text(step), None

    limit = CHAR_LIMITS.get(q_id)
    if limit and len(clean) > limit:
        return (
            f"✂️ Длинновато: {len(clean)} символов при лимите {limit}. "
            "Сократите и отправьте ещё раз.\n\n" + get_question_text(step),
            None,
        )

    # Тяжёлая операция — ВНЕ db.lock (см. docstring).
    masked = mask_pii(clean) if MASK_PII else clean

    def _apply(s):
        # Перечитываем состояние уже под db.lock — на случай гонок.
        st = s.get("step", 0)
        if s.get("finished") or st >= TOTAL_STEPS:
            return None, None
        s["data"][get_question_id(st)] = masked
        new_step = st + 1
        s["step"] = new_step
        if new_step >= TOTAL_STEPS:
            s["finished"] = True
            return None, None
        return get_question_text(new_step), None

    return mutate_session(user_id, _apply)


def is_finished(user_id: str) -> bool:
    return get_session(user_id).get("finished", False)


def get_collected_data(user_id: str) -> dict:
    return get_session(user_id).get("data", {})


def handle_callback(user_id: str, payload: str):
    """Защитный обработчик устаревших кнопок: трактуем payload как текст."""
    return process_answer(user_id, payload)