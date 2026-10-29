"""پیدا کردن قفسه عطر از روی عکس برگه سفارش

اجرا:
    pip install -r requirements.txt
    streamlit run app.py
"""
import html
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from rapidfuzz import fuzz

SHELVES_FILE = Path(__file__).with_name("shelves.json")
MAX_WIDTH = 1600  # عکس بزرگ‌تر از این قبل از OCR کوچک می‌شود
LOW_CONFIDENCE = 90  # زیر این امتیاز، کارت نتیجه هشدار می‌دهد

DEFAULT_SHELVES = {
    "Gucci Guilty": {"shelf": "A3", "aliases": ["گوچی گیلتی", "عطر گوچی گیلتی"]},
    "Dior Sauvage": {"shelf": "B1", "aliases": ["دیور ساواج"]},
    "Chanel Chance": {"shelf": "C2", "aliases": ["شنل چنس"]},
}

# ---------------------------------------------------------------------------
# داده‌ی قفسه‌ها
# ---------------------------------------------------------------------------


def load_shelves() -> dict:
    if SHELVES_FILE.exists():
        try:
            return json.loads(SHELVES_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st.error("فایل shelves.json خراب است؛ از لیست پیش‌فرض استفاده شد.")
    return DEFAULT_SHELVES


def save_shelves(shelves: dict) -> None:
    SHELVES_FILE.write_text(
        json.dumps(shelves, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _clean(v) -> str:
    return "" if pd.isna(v) else str(v).strip()


def shelves_to_df(shelves: dict) -> pd.DataFrame:
    rows = [
        {
            "name": name,
            "aliases": "، ".join(info.get("aliases", [])),
            "shelf": info.get("shelf", ""),
        }
        for name, info in shelves.items()
    ]
    return pd.DataFrame(rows, columns=["name", "aliases", "shelf"])


def df_to_shelves(df: pd.DataFrame) -> dict:
    out = {}
    for _, r in df.iterrows():
        name = _clean(r["name"])
        if not name:
            continue
        aliases = [a.strip() for a in re.split(r"[،,;]", _clean(r["aliases"])) if a.strip()]
        out[name] = {"shelf": _clean(r["shelf"]), "aliases": aliases}
    return out


# ---------------------------------------------------------------------------
# نرمال‌سازی و تطبیق
# ---------------------------------------------------------------------------

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_TO_FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(n) -> str:
    """عدد را با رقم فارسی نشان می‌دهد."""
    return str(n).translate(_TO_FA)


def normalize(text: str) -> str:
    text = text.translate(_DIGITS)
    text = re.sub(r"[ً-ٰٟ]", "", text)  # اعراب
    text = text.replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9؀-ۿ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Hit:
    name: str
    shelf: str
    score: float
    matched_alias: str


def find_hits(lines: list[str], shelves: dict, threshold: int = 80) -> list[Hit]:
    norm_lines = [normalize(l) for l in lines]
    texts = [t for t in norm_lines + [" ".join(norm_lines)] if t]

    hits: list[Hit] = []
    for name, info in shelves.items():
        best, best_alias = 0.0, ""
        for alias in [name] + list(info.get("aliases", [])):
            na = normalize(alias)
            if len(na) < 3:
                continue
            for t in texts:
                # اگر متن کوتاه‌تر از اسم عطر باشد، partial_ratio بی‌معنی بالا می‌رود
                if len(t) < len(na):
                    continue
                s = fuzz.partial_ratio(na, t)
                if s > best:
                    best, best_alias = s, alias
        if best >= threshold:
            hits.append(Hit(name, info.get("shelf", "؟"), best, best_alias))

    # خانواده‌ی اسم‌ها: اگر «Gucci Guilty» داخل «Gucci Guilty Pour Homme» باشد و هر دو match شوند،
    # نام بلندتر فقط وقتی برنده است که امتیازش کمتر از کوتاه‌تر نباشد (با ۳ امتیاز تلورانس)؛
    # وگرنه نام کوتاه‌تر (که امتیاز واضح‌تری دارد) نگه داشته می‌شود.
    nn = {h.name: normalize(h.name) for h in hits}
    drop = set()
    for short in hits:
        for long_ in hits:
            if nn[short.name] != nn[long_.name] and nn[short.name] in nn[long_.name]:
                drop.add(short.name if long_.score >= short.score - 3 else long_.name)
    final = [h for h in hits if h.name not in drop]
    return sorted(final, key=lambda h: -h.score)


def extract_quantity(lines: list[str]):
    text = normalize(" ".join(lines))
    m = re.search(r"تعداد\s*(\d+)", text) or re.search(r"(\d+)\s*تعداد", text)
    return int(m.group(1)) if m else None


def search_manual(query: str, shelves: dict, limit: int = 5):
    q = normalize(query)
    if not q:
        return []
    scored = []
    for name, info in shelves.items():
        s = max(
            fuzz.WRatio(q, normalize(a)) for a in [name] + list(info.get("aliases", []))
        )
        scored.append((s, name, info.get("shelf", "؟")))
    return sorted(scored, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="در حال بارگذاری مدل OCR (بار اول چند دقیقه طول می‌کشد)...")
def get_reader():
    import easyocr

    return easyocr.Reader(["fa", "en"], gpu=False)


def prepare_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if img.width > MAX_WIDTH:
        ratio = MAX_WIDTH / img.width
        img = img.resize((MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    return img


@st.cache_data(show_spinner=False)
def run_ocr(data: bytes) -> list[str]:
    img = prepare_image(data)
    return get_reader().readtext(np.array(img), detail=0, paragraph=False)


# ---------------------------------------------------------------------------
# ظاهر
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&family=Cormorant+Garamond:wght@600;700&display=swap');

:root {
  --mist: #1E1B20;
  --paper: #2A262C;
  --ink: #F1EDF0;
  --ink-soft: #B5ADB6;
  --pink: #C23B78;
  --pink-soft: #F7A8C8;
  --warn: #FFB3D1;
  --line: #4A434B;
  --sans: 'Vazirmatn', Tahoma, sans-serif;
  --serif: 'Cormorant Garamond', 'Vazirmatn', Georgia, serif;
}

.stApp { background: var(--mist); color: var(--ink); }
#MainMenu, footer, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
.block-container { max-width: 680px; padding: 1.6rem 1.1rem 4rem; direction: rtl; }

.stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp li, .stTabs [data-baseweb="tab"] { font-family: var(--sans); }

/* سربرگ */
.head h1 { font-family: var(--sans); font-weight: 800; font-size: 2rem; line-height: 1.3;
  margin: 0; padding: 0; color: var(--ink); }
.head p { margin: .2rem 0 1.1rem; color: var(--ink-soft); font-size: 1rem; }

/* تب‌ها */
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .65rem 0; font-weight: 600; color: var(--ink-soft); background: none; }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--pink-soft); height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* آپلود */
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #6C636D; background: #25212A;
  border-radius: 10px; padding: 1.5rem 1rem; }
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"] span:first-child { font-size: 0; }
[data-testid="stFileUploaderDropzoneInstructions"] span:first-child::after {
  content: "عکس برگه را اینجا رها کن"; font-size: 1rem; font-family: var(--sans); color: var(--ink); }
[data-testid="stFileUploaderDropzone"] button { font-size: 0; border-radius: 10px;
  border: 1.5px solid var(--pink-soft); background: transparent; }
[data-testid="stFileUploaderDropzone"] button::after { content: "انتخاب عکس"; font-size: .95rem;
  font-family: var(--sans); font-weight: 600; color: var(--pink-soft); }

/* ورودی‌ها و دکمه‌ها */
.stTextInput input { border-radius: 10px; font-size: 1.1rem; padding: .7rem .9rem; }
.stButton button[kind="primary"], .stButton button[data-testid="stBaseButton-primary"] {
  background: var(--pink); border: none; color: #fff; border-radius: 10px; font-weight: 600;
  padding: .55rem 1.4rem; }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 10px; background: #25212A; }
[data-testid="stImage"] img { border: 1px solid var(--line); border-radius: 6px; }
[data-testid="stSlider"] { direction: ltr; }
[data-testid="stSlider"] [data-testid="stWidgetLabel"] { direction: rtl; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { direction: ltr; }

/* برچسب قفسه */
.tag { display: flex; align-items: stretch; background: var(--paper); border: 1px solid var(--line);
  border-radius: 6px; overflow: hidden; margin: 0 0 .9rem; box-shadow: 0 2px 0 var(--line); }
.tag-code { background: var(--pink-soft); color: #2B272C; min-width: 7rem; display: flex;
  flex-direction: column; align-items: center; justify-content: center; padding: 1rem .8rem; }
.tag-code b { font-family: var(--sans); font-weight: 800; font-size: 3rem; line-height: 1;
  letter-spacing: .02em; }
.tag-code span { font-size: .85rem; margin-top: .4rem; color: #4A2536; }
.tag-info { flex: 1; padding: .9rem 1.1rem; display: flex; flex-direction: column;
  justify-content: center; gap: .55rem; min-width: 0; }
.tag-name { font-family: var(--serif); font-weight: 700; font-size: 1.75rem; line-height: 1.15;
  color: var(--ink); text-align: right; overflow-wrap: anywhere; }
.tag-meta { display: flex; flex-wrap: wrap; gap: .4rem; }
.chip { font-size: .85rem; padding: .12rem .65rem; border-radius: 999px; border: 1px solid var(--line);
  background: #332E35; color: var(--ink-soft); }
.chip--multi { background: var(--pink-soft); border-color: #E27AA6; color: #2B272C; font-weight: 700; }
.tag-warn { font-size: .9rem; color: var(--warn); line-height: 1.7; }
.tag--low { border: 2px dashed var(--pink-soft); }
.tag--compact .tag-code { min-width: 4.8rem; padding: .7rem .5rem; }
.tag--compact .tag-code b { font-size: 2rem; }
.tag--compact .tag-name { font-size: 1.4rem; }

/* حالت خالی و خطا */
.note { border: 2px dashed #6C636D; border-radius: 6px; padding: 1rem 1.2rem; color: var(--ink-soft);
  line-height: 1.9; }
.note strong { color: var(--ink); display: block; font-size: 1.05rem; }

.count { color: var(--ink-soft); margin: .9rem 0 .4rem; }
</style>
"""


def tag_html(name, shelf, score=None, qty=None, compact=False) -> str:
    low = score is not None and score < LOW_CONFIDENCE
    cls = "tag" + (" tag--low" if low else "") + (" tag--compact" if compact else "")
    parts = [
        f'<div class="{cls}">',
        f'<div class="tag-code"><b dir="ltr">{html.escape(str(shelf))}</b><span>قفسه</span></div>',
        '<div class="tag-info">',
        f'<div class="tag-name" dir="auto">{html.escape(str(name))}</div>',
    ]
    chips = []
    if qty:
        chips.append(f'<span class="chip{" chip--multi" if qty > 1 else ""}">تعداد {fa(qty)}</span>')
    if score is not None:
        chips.append(f'<span class="chip">اطمینان {fa(round(score))}٪</span>')
    if chips:
        parts.append('<div class="tag-meta">' + "".join(chips) + "</div>")
    if low:
        parts.append(
            '<div class="tag-warn">اطمینان کم است؛ اسم روی برگه را با این عطر مقایسه کن.</div>'
        )
    parts.append("</div></div>")
    return "".join(parts)


def note_html(title: str, body: str) -> str:
    return f'<div class="note"><strong>{html.escape(title)}</strong>{html.escape(body)}</div>'


# ---------------------------------------------------------------------------
# تب‌ها
# ---------------------------------------------------------------------------


def scan_tab(shelves: dict):
    source = st.radio(
        "منبع عکس", ["آپلود عکس", "دوربین"], horizontal=True, label_visibility="collapsed"
    )

    data = None
    if source == "آپلود عکس":
        f = st.file_uploader(
            "عکس برگه سفارش", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
        )
        if f:
            data = f.getvalue()
    else:
        c = st.camera_input("از برگه عکس بگیر", label_visibility="collapsed")
        if c:
            data = c.getvalue()

    if data is None:
        return

    with st.spinner("در حال خواندن برگه…"):
        lines = run_ocr(data)

    threshold = st.session_state.get("threshold", 80)
    hits = find_hits(lines, shelves, threshold)
    qty = extract_quantity(lines)

    if hits:
        for h in hits:
            st.markdown(tag_html(h.name, h.shelf, score=h.score, qty=qty), unsafe_allow_html=True)
    else:
        st.markdown(
            note_html(
                "عطری پیدا نشد",
                "سخت‌گیری تطبیق را کمتر کن، یا اسم عطر را در تب «جستجو» تایپ کن. "
                "اگر عطر در لیست نیست، از تب «قفسه‌ها» اضافه‌اش کن.",
            ),
            unsafe_allow_html=True,
        )

    with st.expander("عکس برگه و متن خوانده‌شده"):
        try:
            st.image(prepare_image(data), width="stretch")
        except Exception:  # نسخه‌های قدیمی‌تر Streamlit
            st.image(prepare_image(data), use_container_width=True)
        st.code("\n".join(lines) or "(متنی خوانده نشد)", language=None)

    with st.expander("تنظیمات تطبیق"):
        st.slider(
            "سخت‌گیری تطبیق", 60, 100, 80, key="threshold",
            help="اگر عطر پیدا نشد عدد را کم کن؛ اگر عطر اشتباه نشان داد زیاد کن.",
        )


def manual_tab(shelves: dict):
    q = st.text_input(
        "نام عطر",
        placeholder="مثلاً گوچی گیلتی یا Gucci Guilty",
        label_visibility="collapsed",
    )
    if not q.strip():
        return
    results = [r for r in search_manual(q, shelves) if r[0] >= 60]
    if results:
        for _, name, shelf in results:
            st.markdown(tag_html(name, shelf, compact=True), unsafe_allow_html=True)
    else:
        st.markdown(
            note_html("نتیجه‌ای پیدا نشد", "املای دیگری امتحان کن، یا عطر را از تب «قفسه‌ها» اضافه کن."),
            unsafe_allow_html=True,
        )


def manage_tab(shelves: dict):
    if st.session_state.pop("just_saved", False):
        st.success("لیست ذخیره شد.")

    st.markdown(f'<div class="count">{fa(len(shelves))} عطر در لیست</div>', unsafe_allow_html=True)
    st.caption(
        "اسم انگلیسی را در ستون اول و اسم فارسی را در ستون دوم بنویس (چند اسم را با ویرگول جدا کن)."
    )

    cfg = {
        "name": st.column_config.TextColumn("نام عطر", required=True),
        "aliases": st.column_config.TextColumn("نام‌های دیگر"),
        "shelf": st.column_config.TextColumn("قفسه", required=True, width="small"),
    }
    try:
        edited = st.data_editor(
            shelves_to_df(shelves), num_rows="dynamic", width="stretch",
            column_config=cfg, key="editor",
        )
    except Exception:  # نسخه‌های قدیمی‌تر Streamlit
        edited = st.data_editor(
            shelves_to_df(shelves), num_rows="dynamic", use_container_width=True,
            column_config=cfg, key="editor",
        )

    if st.button("ذخیره لیست", type="primary"):
        save_shelves(df_to_shelves(edited))
        st.session_state["just_saved"] = True
        st.rerun()


def main():
    st.set_page_config(page_title="قفسه عطر", page_icon="🧴", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="head"><h1>قفسه عطر</h1>'
        "<p>عکس برگه‌ی سفارش را بده تا قفسه‌ی هر عطر را ببینی.</p></div>",
        unsafe_allow_html=True,
    )

    shelves = load_shelves()
    t1, t2, t3 = st.tabs(["اسکن برگه", "جستجو", "قفسه‌ها"])
    with t1:
        scan_tab(shelves)
    with t2:
        manual_tab(shelves)
    with t3:
        manage_tab(shelves)


if __name__ == "__main__":
    main()
