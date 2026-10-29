"""پیدا کردن قفسه عطر از روی عکس برگه سفارش

اجرا:
    pip install -r requirements.txt
    streamlit run app.py
"""
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


def normalize(text: str) -> str:
    text = text.translate(_DIGITS)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)  # اعراب
    text = text.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u0600-\u06FF ]+", " ", text)
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


@st.cache_resource(show_spinner="در حال بارگذاری مدل OCR (بار اول کمی طول می‌کشد)...")
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
# رابط کاربری
# ---------------------------------------------------------------------------


def scan_tab(shelves: dict):
    threshold = st.slider(
        "حساسیت تطبیق (کمتر = آسان‌گیرتر)", 60, 100, 80,
        help="اگر عطر پیدا نمی‌شود عدد را کمتر کن؛ اگر عطرهای اشتباه نشان می‌دهد بیشتر کن.",
    )
    source = st.radio("منبع عکس", ["آپلود فایل", "دوربین"], horizontal=True)

    data = None
    if source == "آپلود فایل":
        f = st.file_uploader("عکس برگه سفارش", type=["jpg", "jpeg", "png", "webp"])
        if f:
            data = f.getvalue()
    else:
        c = st.camera_input("از برگه عکس بگیر")
        if c:
            data = c.getvalue()

    if data is None:
        return

    st.image(prepare_image(data), width=380)
    with st.spinner("در حال خواندن برگه..."):
        lines = run_ocr(data)

    with st.expander("متن خوانده‌شده توسط OCR"):
        st.write(lines)

    hits = find_hits(lines, shelves, threshold)
    qty = extract_quantity(lines)

    if hits:
        for h in hits:
            st.success(f"### {h.name}\n\n## 📍 قفسه: {h.shelf}\n\nاطمینان: {h.score:.0f}٪")
        if qty:
            st.info(f"تعداد ثبت‌شده در برگه: {qty}")
    else:
        st.error("هیچ عطری پیدا نشد. متن خوانده‌شده را بررسی کن یا پایین جستجوی دستی بزن.")


def manual_tab(shelves: dict):
    q = st.text_input("اسم عطر را تایپ کن (فارسی یا انگلیسی)")
    for score, name, shelf in search_manual(q, shelves):
        st.write(f"**{name}** ← قفسه **{shelf}**  ({score:.0f}٪)")


def manage_tab(shelves: dict):
    st.caption(
        "name: اسم اصلی (ترجیحاً انگلیسی) • aliases: اسم‌های دیگر با ویرگول جدا شود "
        "(مثلاً نام فارسی) • shelf: شماره قفسه"
    )
    edited = st.data_editor(
        shelves_to_df(shelves), num_rows="dynamic", use_container_width=True, key="editor"
    )
    if st.button("ذخیره تغییرات", type="primary"):
        save_shelves(df_to_shelves(edited))
        st.success("ذخیره شد.")
        st.rerun()


def main():
    st.set_page_config(page_title="قفسه عطر", page_icon="🧴")
    st.markdown(
        "<style>.stMarkdown, .stAlert {direction: rtl; text-align: right;}</style>",
        unsafe_allow_html=True,
    )
    st.title("🧴 پیدا کردن قفسه عطر")

    shelves = load_shelves()
    t1, t2, t3 = st.tabs(["اسکن برگه", "جستجوی دستی", "مدیریت قفسه‌ها"])
    with t1:
        scan_tab(shelves)
    with t2:
        manual_tab(shelves)
    with t3:
        manage_tab(shelves)


if __name__ == "__main__":
    main()
