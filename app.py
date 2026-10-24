"""Driver planner for the charity's coordinators. Runs in the browser (stlite) or with `streamlit run app/app.py`.
Models: planner.py. Upload checks: checks.py and convert.py.

Written for a non-technical team: plain words, no service codes, everything important on the page rather than in
hover text, and colour never the only way something is shown."""
import hashlib
import io
import math
from pathlib import Path

import pandas as pd
import streamlit as st

import checks
import ui
from planner import (HOSPITAL_NAMES, MILES_PER_KM, SERVICE_NAMES, WEEKDAYS, Dropout, Network, Settings, coverage,
                     expansion, expansion_table, load_public, trainees)

HERE = Path(__file__).parent
DEMO = HERE / "demo_data"
FILES = {"drivers": "Drivers list", "availability": "Driver availability", "journeys": "Journeys"}

st.set_page_config(page_title="Driver planner", page_icon="🚗", layout="wide")
ui.style()
ui.banner()


@st.cache_resource
def public():
    return load_public(HERE / "public_data")


def read(blob):
    """A CSV as Excel saves it: UTF-8 or Windows encoding, commas or semicolons. None if it can't be read."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(blob), sep=None, engine="python", encoding=enc, dtype=str)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def prepare(kind, blobs):
    """Read, tidy and check the uploaded files. kind is "csv" (blobs = the three CSVs, then the mileage CSV or
    None) or "xlsx" (blobs = (file name, bytes) pairs of the charity's own workbooks).
    Returns (tables, errors, warnings, notes)."""
    notes = []
    if kind == "xlsx":
        import convert
        try:
            tables, notes = convert.from_workbooks(list(blobs))
        except convert.ConvertError as e:
            return {}, [(str(e), None)], [], []
    else:
        tables = {}
        for f, blob in zip([*FILES, "mileage"], blobs):
            if blob is None:
                continue
            df = read(blob)
            if df is None:
                label = FILES.get(f, "mileage claims").lower()
                return {}, [(f"The {label} file couldn't be opened. Save it from Excel with File, Save As, and "
                             "choose 'CSV UTF-8'.", None)], [], []
            tables[f] = df
    for f in list(tables):
        tables[f], tidied = checks.tidy(f, tables[f])
        notes += tidied
    errors, warnings = checks.validate({f: tables[f] for f in FILES if f in tables}, public()["coords"])
    return tables, errors, warnings, notes


@st.cache_resource(show_spinner=False)
def build(sig, _tables):
    """The model, built once per set of files (sig is a fingerprint of them)."""
    net = Network(_tables["drivers"], _tables["availability"], _tables["journeys"], public()["coords"], Settings())
    return net, Dropout(net)


@st.cache_resource(show_spinner=False)
def remembered(sig, what, arg, _net, _drop):
    """Slow results, worked out once per set of files and remembered between clicks."""
    if what == "coverage":
        return coverage(_net)
    if what == "ranking":
        return _drop.ranking()
    if what == "turnover":
        return _drop.turnover(arg, draws=200)
    return expansion_table(_net, public(), arg)


STATUS_ORDER = ["No one free", "Not every week", "Relies on one driver", "Covered"]
# light backgrounds with dark text; the words carry the meaning, the colour only helps scanning
STATUS_COLOURS = {"No one free": "#F5C6CB", "Not every week": "#FAD7B5", "Relies on one driver": "#FFEEBA",
                  "Covered": "#C3E6CB", "No journeys": "#F2F2F2"}


@st.cache_resource
def district_centres():
    """Centre of each postcode district (G51, KA1...), from every live postcode in it."""
    c = public()["coords"]
    return c.groupby(c.index.str[:-3])[["lat", "lon"]].mean()


def file_guide(expanded):
    with st.expander("What the files should look like", expanded=expanded):
        st.markdown("Each file needs these column names in its first row, spelled exactly as shown. Other columns "
                    "are ignored. Download a template to see the layout, then copy your rows into it.")
        for f, label in {**FILES, "mileage": "Mileage claims, optional"}.items():
            st.markdown(f"**{label} ({f}.csv)**")
            ui.table(checks.guide(f), key=f"guide_{f}")
            st.download_button(f"Download {f}.csv template", checks.template(f), f"{f}_template.csv", key=f"t_{f}")


def up(x):
    """Rounded up to a whole number: a figure that's easy to read and hear, and never shows real demand as 0."""
    return math.ceil(round(float(x), 6))


def shares(values):
    """Whole numbers that add up to the rounded-up total: each part rounded down, then the rest go to the parts with
    the biggest remainders. So a headline, its parts and the comparison table all agree."""
    values = [float(v) for v in values]
    whole = [math.floor(round(v, 6)) for v in values]
    for i in sorted(range(len(values)), key=lambda i: values[i] - whole[i], reverse=True)[:up(sum(values)) - sum(whole)]:
        whole[i] += 1
    return whole


def money(x):
    return "not known" if x != x else f"£{max(x, 0):,.0f}"


def services(codes):
    return ", ".join(SERVICE_NAMES.get(c, c) for c in codes.split("+"))


# ---------- sidebar: the charity's files ----------
with st.sidebar:
    st.header("Use newer data")
    st.markdown("Your files stay on this computer and are never sent anywhere.")
    mode = st.radio("What will you add?", ["The charity's spreadsheets, as they are", "Three CSV files in the planner's layout"])
    kind, blobs = None, None
    if mode.startswith("The charity's"):
        books = st.file_uploader("The journeys, availability and driver details workbooks (.xlsx). Add the "
                                 "mileage report too if you'd like mileage figures.", type=["xlsx", "csv"],
                                 accept_multiple_files=True, key="books")
        if books:
            kind, blobs = "xlsx", tuple((b.name, b.getvalue()) for b in books)
    else:
        st.markdown("'What the files should look like', under Your data, shows the columns each one needs.")
        uploads = {f: st.file_uploader(f"{label} ({f}.csv)", type="csv", key=f) for f, label in FILES.items()}
        mileage_upload = st.file_uploader("Mileage claims, optional (mileage.csv)", type="csv", key="mileage")
        added = sum(u is not None for u in uploads.values())
        if added == len(FILES):
            kind = "csv"
            blobs = (*(uploads[f].getvalue() for f in FILES), mileage_upload.getvalue() if mileage_upload else None)
        elif added:
            st.warning(f"You've added {added} of 3 files. Add the other {len(FILES) - added} to use your own data.")
    own = kind is not None
    if not own and DEMO.exists():
        kind = "csv"
        blobs = tuple((DEMO / f"{f}.csv").read_bytes() for f in FILES) + ((DEMO / "mileage.csv").read_bytes()
                                                                         if (DEMO / "mileage.csv").exists() else None,)
    if own:
        st.success("Using your files.")

if kind is None:
    ui.how_to_use()
    file_guide(expanded=False)
    st.stop()

with st.spinner("Reading and checking the files…"):
    tables, errors, warnings, notes = prepare(kind, blobs)
if errors:
    st.error("The planner can't use these files yet. Fix the problems below in the spreadsheet and add the files "
             "again.")
    for i, (msg, rows) in enumerate(errors):
        st.markdown(f"- {msg}")
        if rows is not None:
            ui.table(rows, key=f"error_{i}")
    file_guide(expanded=kind == "csv")
    st.stop()

sig = hashlib.md5(repr(blobs).encode()).hexdigest()
try:
    with st.spinner("Reading and checking the files…"):
        net, drop = build(sig, tables)
except Exception as e:  # anything the checks missed should still explain itself
    st.error("Something in the files stopped the planner, and the checks didn't catch what. Compare each file "
             "with 'What the files should look like' below, especially the dates, names and yes/no columns.")
    with st.expander("Details to pass on if you ask for help"):
        st.code(str(e))
    file_guide(expanded=True)
    st.stop()
pub = public()
mileage = tables.get("mileage")

if not own:
    st.info("You're looking at the sample weeks the charity shared in 2026. Add your own files in the panel on "
            "the left to see your latest weeks.")
weeks = pd.to_datetime(net.driver_weeks)
if len(weeks) <= 2:
    st.markdown(f"Based on the week of {' and '.join(weeks.strftime('%-d %B %Y'))}, the latest week that records "
                "who drove each journey.")
else:
    st.markdown(f"Based on {len(weeks)} weeks that record who drove each journey, from the week of "
                f"{weeks[0]:%-d %B %Y} to the week of {weeks[-1]:%-d %B %Y}.")

# ---------- pages ----------
def page_overview():
    import overview
    overview.render_overview(net, district_centres(), HERE / "public_data")

def page_recruit():
    import overview
    overview.render_recruit(net, pd.read_csv(HERE / "public_data" / "districts.csv"))

def page_short():
    with st.spinner("Working out where cover is thin…"):
        cells, rows, free = remembered(sig, "coverage", None, net, drop)
    cells, rows = cells.copy(), rows.copy()
    label = lambda r: f"{SERVICE_NAMES.get(r['service'], r['service'])} at {HOSPITAL_NAMES.get(r['hospital'], r['hospital'])}"
    rows["name"] = rows.apply(label, axis=1)
    cells["name"] = cells.apply(label, axis=1)
    weak = rows[rows["train_days"].map(len) > 0]

    st.header("Where cover is thin")
    if weak.empty:
        st.markdown("Every service has at least two trained drivers free on the days it runs.")
    for r in weak.itertuples():
        parts = []
        if r.top_share >= 0.5:
            parts.append(f"one driver did {r.top_share:.0%} of its volunteer journeys")
        by_status = cells[(cells["name"] == r.name) & cells["status"].isin(STATUS_ORDER[:3])]
        for status in STATUS_ORDER[:3]:
            days = by_status.loc[by_status["status"] == status, "weekday"].tolist()
            if days:
                parts.append(f"{' and '.join(days)}: {status.lower()}")
        st.markdown(f"- **{r.name}** ({r.journeys_pw:.0f} journeys a week): {'; '.join(parts)}.")

    grid = cells.pivot(index="name", columns="weekday", values="status").reindex(
        columns=[d for d in WEEKDAYS if (cells.loc[cells["weekday"] == d, "status"] != "").any()]).fillna("")
    grid = grid.replace("", "No journeys").rename(columns=lambda d: d[:3]).rename_axis("Service")
    grid.columns.name = None
    ui.table(grid.style.map(lambda v: f"background-color: {STATUS_COLOURS.get(v, '#FFFFFF')}; color: #1A1A1A"),
             key="coverage_grid", index=True)

    st.header("Who to ask about training")
    if weak.empty:
        st.markdown("No training needed to fill a gap right now.")
    else:
        pick = st.selectbox("Service", weak["name"])
        r = weak.set_index("name").loc[pick].to_dict()
        t = trainees(net, r, free)
        why = "on the days it's short" if r["weak_days"] else "to share the work, since one driver does most of it"
        if t.empty:
            st.warning(f"Nobody who lives nearby and isn't already trained is free on {', '.join(r['train_days'])}. "
                       "This gap needs new drivers near the hospital, or asking current drivers about those days.")
        else:
            st.markdown(f"Drivers not yet trained for {pick.split(' at ')[0].lower()}, who live nearby and are "
                        f"free {why} ({', '.join(r['train_days'])}). Their notes are shown so you can check "
                        "they'd be willing.")
            ui.table(pd.DataFrame({
                "Driver": t["driver"], "Area": t["district"], "Distance to hospital (miles)": (t["km"] * MILES_PER_KM).round(0).astype(int),
                "Free on": t["days"], "Already trained for": t["trained_for"].map(
                    lambda x: ", ".join(SERVICE_NAMES.get(c, c) for c in x.split(", ") if c)),
                "Notes": t["notes"]}), key="trainees")

    blank = (net.availability["status"].astype(str).str.lower() == "unknown").mean()
    st.markdown(f"**About the data:** {blank:.0%} of the availability sheet is blank. A blank day counts as not "
                "free unless the driver drove that day, so some gaps may be smaller than shown. Filling in the "
                "sheet makes this page more accurate.")

    with st.expander("How this is worked out"):
        st.markdown(f"""
For each service at each hospital we count, for every day of the week, how many drivers trained for it, living
within {net.hospital_reach_km * MILES_PER_KM:.0f} miles of the hospital, are free on a typical week. "Relies on one driver" means
only one is usually free; "Not every week" means fewer than one; "No one free" means none in any week we have.
Journeys come from all the sample weeks; who's free comes from every week of the availability sheet.

Drivers to train are those who live nearby, aren't trained for that service yet and are free on the days that
need cover, most often free first.
""")

def page_driver_stops():
    busiest = drop.legs["driver"].value_counts().index[0]
    leavers = st.multiselect("Which drivers stop? You can pick more than one.", drop.active, default=[busiest])
    if leavers:
        out, log = drop.replay(set(leavers), detail=True)
        done = drop.legs["driver"].isin(leavers).sum() * drop.per_year / 12
        c = st.columns(4)
        c[0].metric("Journeys a month they drive", f"{done:,.0f}")
        c[1].metric("Another volunteer could cover", f"{out['journeys_covered_py'] / 12:,.0f}")
        c[2].metric("Would need a taxi", f"{out['journeys_to_taxi_py'] / 12:,.0f}")
        c[3].metric("Extra cost a month", money(out["extra_cost_py"] / 12))

        st.header("Who could cover each day")
        days = pd.DataFrame({
            "Driver who stops": log["driver_leaving"],
            "Day": pd.to_datetime(log["date"]).dt.strftime("%A %-d %B"),
            "Services": log["services"].map(services),
            "Journeys": log["journeys"],
            "Covered by": log["covered_by"].replace({"taxi": "Nobody free: taxi"}),
            "Extra cost that day": log["extra_cost"].map(money),
        })
        ui.table(days, key="cover_days")

    st.header("Whose absence would be hardest to cover")
    with st.spinner("Working out the cost for each driver…"):
        ranking = remembered(sig, "ranking", None, net, drop)
    view = pd.DataFrame({
        "Driver": ranking.index,
        "Services": ranking["services"].map(services),
        "Journeys a month": (ranking["journeys_py"] / 12).round(0).astype(int),
        "Would need a taxi": (ranking["journeys_to_taxi_py"] / 12).round(0).astype(int),
        "Extra cost a month (£)": (ranking["extra_cost_py"] / 12).clip(lower=0).round(0),
    })
    ui.table(view, key="hardest", formats={"Extra cost a month (£)": "£{:,.0f}"}, column_config={
        "Extra cost a month (£)": st.column_config.ProgressColumn(
            format="£%.0f", min_value=0, max_value=float(view["Extra cost a month (£)"].max()))})
    st.markdown("Long milk bank runs to Dundee and Aberdeen usually go to partner charities rather than taxis, so "
                "the cost of losing a driver who mostly does those runs is likely lower than shown.")
    st.download_button("Download this table (opens in Excel)", view.to_csv(index=False).encode(),
                       "hardest_to_cover.csv")

    st.header("If several drivers stop volunteering this year")
    pct = st.slider("Percentage of this week's drivers who stop", 5, 50, 20, 5, format="%d%%")
    with st.spinner("Trying out many groups of drivers stopping…"):
        k, runs = remembered(sig, "turnover", pct / 100, net, drop)
    c = st.columns(3)
    c[0].metric("Drivers who stop", k)
    c[1].metric("Likely extra cost a month", money(runs["extra_cost_py"].median() / 12))
    c[2].metric("Extra cost if key drivers are among them", money(runs["extra_cost_py"].quantile(0.9) / 12))

    with st.expander("How this is worked out"):
        st.markdown(f"""
For each day a leaving driver drove, we look for another volunteer who was free that day, is trained for the same
services and lives within {net.pickup_reach_km * MILES_PER_KM:.0f} miles of the first pickup. The nearest one takes the day. If
nobody fits, that day's journeys go by taxi at Glasgow fares. The extra cost is the taxi fares minus the mileage
the charity no longer pays, or the extra miles a stand-in drives.

Costs are shown a month at a time because replacing a driver takes about a month: roughly two weeks to find
someone and two weeks to train them. So a month's extra cost is about what one driver stopping costs.

Things this leaves out:

- Drivers whose availability is blank count as not free, so the cost may be a little high.
- The cost of recruiting and training a replacement.
- Journeys that don't happen at all, and missed appointments.
""")

def page_journey():
    import match
    match.render(net, district_centres())

def page_mileage():
    problems = checks.validate_mileage(mileage)
    if problems:
        st.error("The mileage file can't be used yet. Fix the problems below in the spreadsheet, export again and "
                 "add it once more. The other questions still work.")
        for i, (msg, rows) in enumerate(problems):
            st.markdown(f"- {msg}")
            if rows is not None:
                ui.table(rows, key=f"mileage_problem_{i}")
    else:
        import mileage as mileage_page
        mileage_page.render(net, drop, mileage)

def page_rota():
    import rota
    rota.render(net)

def page_new_department():
    estimates = {"central": "Middle (our best guess)", "low": "Lower", "high": "Higher"}
    scenario = st.radio("How busy will patient transport be?", list(estimates), format_func=estimates.get,
                        horizontal=True)
    with st.spinner("Working out each department…"):
        table = remembered(sig, "expansion", scenario, net, drop)
    show_served = st.checkbox("Also list maternity units the charity already serves", value=False)
    options = table if show_served else table[~table["already_served"]]
    ids = pub["candidates"].set_index("label")["candidate"]
    choice = st.selectbox("Department or maternity unit (busiest first)", options["candidate"])
    r = expansion(net, pub, ids[choice], scenario)

    st.header(choice)
    if r["already_served"]:
        st.info("The charity already serves this unit. The figures show the work it brings today.")
    # whole numbers: the total rounded up, shared out so the parts add up to it
    split = r["by_service"]
    parts = shares(split[f"journeys_pw_{scenario}"])
    st.markdown(f"**About {sum(parts)} extra journeys a week**, made up of:")
    c = st.columns(len(split))
    for col, service, n in zip(c, split["service"], parts):
        col.metric(SERVICE_NAMES.get(service, service), f"{n} a week")
    pat = split[split["service"] == "PAT"]
    if len(pat):
        p = pat.iloc[0]
        st.markdown(f"Patient transport could be anywhere from {up(p['journeys_pw_low'])} to "
                    f"{up(p['journeys_pw_high'])} journeys a week.")

    need = up(r["drivers_needed"])
    st.markdown(f"**Drivers needed:** up to {need} driver{'s' if need != 1 else ''}' worth of time, made up of:")
    c = st.columns(3)
    c[0].metric("Current drivers with spare days", len(r["from_spare_days"]))
    c[1].metric("Current drivers to train", len(r["to_train"]))
    c[2].metric("New drivers to recruit", r["to_recruit"])
    lines = []
    if r["from_spare_days"]:
        lines.append(f"- **Already trained and free:** {', '.join(r['from_spare_days'])}")
    if r["to_train"]:
        lines.append(f"- **Free and nearby, but need training:** {', '.join(r['to_train'])}")
    if r["to_recruit"]:
        lines.append(f"- **Recruit in:** {', '.join(r['recruit_districts'])}. These postcode areas are near the "
                     "hospital and have the most families without a car.")
    st.markdown("\n".join(lines))

    st.header("Compare all departments and maternity units")
    compare = pd.DataFrame({
        "Department or unit": options["candidate"],
        "Area": options["health_board"],
        "Extra journeys a week": options[f"journeys_pw_{scenario}"].map(up),
        "Current drivers with spare days": options["from_spare_days"],
        "Current drivers to train": options["to_train"],
        "New drivers to recruit": options["to_recruit"],
        "Recruit in": options["recruit_districts"],
    })
    ui.table(compare, key="compare")
    st.download_button("Download this table (opens in Excel)", compare.to_csv(index=False).encode(),
                       "new_departments.csv")

    with st.expander("How this is worked out"):
        st.markdown(f"""
We take how many journeys the charity does today for each patient (patient transport, respiratory) or each birth
(milk bank, breast pumps, neonatal), and multiply by the department's patients or the unit's births from Public
Health Scotland ({pub['constants']['source_year']}). Areas where more families have no car get more journeys.

A volunteer does about {round(net.journeys_per_day)} journeys on a driving day. The extra days are filled first by
trained drivers with spare days who live within {net.hospital_reach_km * MILES_PER_KM:.0f} miles of the hospital, then by nearby
drivers who would need training, and the rest by new recruits.

Things this leaves out:

- Each department is looked at on its own, as if it were the only one added.
- Memory making, cardiology and the BIOPIC study aren't counted, because they don't grow with patients or births.
- The cost of recruiting and training.

Figures on this page are rounded up to whole numbers, so they never understate the work or the drivers needed.
""")



def page_files():
    with st.expander("How to use this"):
        st.markdown(f"Add new files in the panel on the left at any time; every page updates. Step-by-step "
                    f"instructions, with pictures, are in {ui.GUIDE}.")
    st.header("How the files were read")
    st.markdown("\n".join(f"- {line}" for line in notes) if notes else "The files were read as they are.")
    st.header("Things to check in the files")
    if not warnings:
        st.markdown("Nothing to check: the files look complete.")
    else:
        st.markdown(f"{len(warnings)} things to check. The planner still works.")
    for i, (msg, rows) in enumerate(warnings):
        st.markdown(f"- {msg}")
        if rows is not None:
            ui.table(rows, key=f"warning_{i}")
    file_guide(expanded=True)


# ---------- the menu ----------
def page(fn, title, url, default=False):
    def show():   # the tab title names the page, so a screen reader announces it
        try:
            st.set_page_config(page_title=f"{title} · Driver planner")
        except st.errors.StreamlitAPIException:   # Streamlit before 1.37 allows only one call; the tab keeps the first title
            pass
        fn()
    return st.Page(show, title=title, url_path=url, default=default)


st.navigation({
    "See how things are": [page(page_overview, "Overview", "overview", default=True),
                           *([page(page_mileage, "Mileage and claims", "mileage")] if mileage is not None else [])],
    "This week": [page(page_journey, "Who could do this journey?", "journey"),
                  page(page_rota, "Draft next week's rota", "rota")],
    "Plan ahead": [page(page_short, "Where are we short?", "short"),
                   page(page_driver_stops, "What if a driver stops?", "driver-stops"),
                   page(page_new_department, "What if we add a department?", "new-department"),
                   # demo page only: the recruitment view (overview.py)
                   *([page(page_recruit, "Where should we recruit?", "recruit")] if DEMO.exists() else [])],
    "Your data": [page(page_files, "Files: checks and layout", "files")],
}, position="sidebar").run()
