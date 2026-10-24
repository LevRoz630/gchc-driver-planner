"""Shared page pieces: the banner, the contrast fixes, and tables a screen reader can read.

st.dataframe draws on a canvas, which screen readers can't read. Short tables are shown as real HTML tables
(st.table); long ones keep the sortable view, with a switch above them to show the same rows as a plain table."""
import inspect

import pandas as pd
import streamlit as st

# older Streamlit (before 1.56 / 1.50) has no st.table(hide_index=) or st.dataframe(width="stretch")
TABLE_HIDES_INDEX = "hide_index" in inspect.signature(st.table).parameters
STRETCH = ({"width": "stretch"} if tuple(int(n) for n in st.__version__.split(".")[:2]) >= (1, 50)
           else {"use_container_width": True})

LONG_ROWS = 15   # longer than this, a plain table gets hard to scan, so it starts sortable


def style():
    # Streamlit draws uploaded-file sizes, the upload box's size limit, message-box text and table column headings
    # too light (3.7:1 at best); darken them to pass WCAG AA contrast
    st.markdown("<style>[data-testid='stFileChipName'], [data-testid='stFileChipName'] + div, [data-testid='stAlert'] p, "
                "[data-testid='stFileUploaderDropzoneInstructions'] span, [data-testid='stTableStyledTable'] th, "
                "[data-testid='stTableStyledTable'] th p {color: #1A1A1A !important;}</style>", unsafe_allow_html=True)


def banner():
    # the page's one main heading, in the charity's deep purple (white on #560C70 is 12.5:1)
    st.markdown("<h1 style='background:#560C70;color:#FFFFFF;padding:0.6rem 1rem;border-radius:6px;font-size:1.4rem;"
                "font-weight:600;margin:0'>Glasgow Children's Hospital Charity · Driver planner</h1>",
                unsafe_allow_html=True)


GUIDE = "the \"Using the planner\" section of the report PDF we gave you"


def how_to_use():
    """The steps on one screen, in words (no screenshots), with the full guide in the report PDF."""
    with st.container(border=True):
        st.header("How to use this")
        st.markdown(f"""
1. **Open the panel on the left.** On a phone or a narrow window it starts closed: tap the **»** button at the top
   left.
2. **Keep "The charity's spreadsheets, as they are" selected and add your files:** the journeys, availability and
   driver details workbooks (.xlsx), plus the yearly mileage report if you'd like mileage figures. The file and tab
   names don't matter.
3. **Wait a few seconds.** The menu appears in the panel and the Overview opens with your latest week. The very
   first time, the page takes about half a minute to get ready.
4. **Open "Files: checks and layout" once** to see what was read and anything worth checking in the spreadsheets.
5. **Pick what you need from the menu:** *This week* to find a driver for a journey or draft next week's rota,
   *Plan ahead* for training, recruiting and "what if" questions.

Your files stay on this computer and are never sent anywhere. Step-by-step instructions, with pictures, are in
{GUIDE}.
""")


def styled(df, formats=None):
    """Numbers as people write them: whole numbers without decimals, others to one decimal, blanks left blank."""
    floats = df.select_dtypes("float").columns
    whole = [c for c in floats if (df[c].dropna() % 1 == 0).all()]
    s = df.style.format(na_rep="")
    s = s.format("{:,.0f}", subset=whole, na_rep="").format("{:,.1f}", subset=[c for c in floats if c not in whole],
                                                             na_rep="")
    for col, fmt in (formats or {}).items():
        s = s.format(fmt, subset=[col], na_rep="")
    return s


def table(data, *, key, formats=None, column_config=None, index=False):
    """A table screen readers can read. `data` is a DataFrame or a pandas Styler (kept as given, colours too)."""
    s = styled(data, formats) if isinstance(data, pd.DataFrame) else data
    if len(s.data) <= LONG_ROWS or st.toggle("Show as a plain table (easier for screen readers)", key=f"{key}_plain"):
        if TABLE_HIDES_INDEX:
            st.table(s, hide_index=not index)
        else:
            st.table(s if index else s.hide(axis="index"))
    else:
        st.dataframe(s, hide_index=not index, column_config=column_config, **STRETCH)
