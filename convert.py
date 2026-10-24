"""Parse the charity's workbooks (weekly journeys, availability, mileage report) into tidy tables.

Used by the app on uploaded bytes and by src/clean_data.py on the raw files. Tabs are recognised by their header
text, not by file or sheet name. Runs in the browser (stlite): pandas, numpy and openpyxl only, no disk or network."""
import datetime as dt
import io
import re
from pathlib import Path

import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
POSTCODE = r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d)\s*([A-Z]{2})\b"  # tolerates "G31 2 ER"
OUTWARD_AT_END = r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*$"  # addresses that give only "G74" or "KA13"
TIME = r"(\d{1,2}(?:[.:]\d{2})?\s*[ap]m)"
# names in the volunteer column that aren't volunteers; anything else with "/" is a relay or partner hand-over too.
# HAIBB and NERVS are the partner charities that bring milk from the north to Perth (charity's own sheet notes)
NON_VOLUNTEERS = {"taxi": "taxi", "haibb": "relay_or_partner", "nervs": "relay_or_partner"}
TRAINING = ["PAT", "RESP", "MB", "BP", "NNU"]  # always given a role_ column, even if no one has it

# public addresses, so hospital_locations.csv is optional
HOSPITALS = [
    ("RHC", "Royal Hospital for Children (Glasgow)", "G51 4TF"),
    ("CCF", "Crosshouse Hospital (Kilmarnock)", "KA2 0BE"),
    ("PRM", "Princess Royal Maternity (Glasgow)", "G31 2ER"),
    ("WISHAW", "Wishaw General Hospital", "ML2 0DP"),
    ("RIE", "Royal Infirmary of Edinburgh", "EH16 4SA"),
    ("ARI", "Aberdeen Royal Infirmary", "AB25 2ZN"),
    ("CDS", "Children's Diabetes Service - Blythswood House (Renfrew)", "PA4 8NT"),
    ("RAH", "Royal Alexandra Hospital (Paisley)", "PA2 9PN"),
    ("FVRH", "Forth Valley Royal Hospital (Larbert)", "FK5 4WR"),
    ("RAIGMORE", "Raigmore Hospital (Inverness)", "IV2 3UJ"),
    ("NINEWELLS", "Ninewells Hospital (Dundee)", "DD2 1SY"),  # journeys say "DD1 1GZ", which doesn't exist
]
HOSPITAL_WORDS = {"princess royal maternity": "PRM", "prm ": "PRM", "royal hospital for children": "RHC",
                  "milk bank": "RHC", "crosshouse": "CCF", "wishaw": "WISHAW", "royal infirmary of edinburgh": "RIE",
                  "eri neonatal": "RIE", "aberdeen royal infirmary": "ARI", "aberdeen maternity": "ARI",
                  "blythswood": "CDS", "children's diabetes": "CDS", "royal alexandra": "RAH", "rah ": "RAH",
                  "forth valley": "FVRH", "raigmore": "RAIGMORE", "ninewells": "NINEWELLS"}


class ConvertError(Exception):
    """A problem the coordinator can fix; the message says which file and tab, and what to change."""


def norm_postcode(s):
    m = re.search(POSTCODE, str(s).upper())
    return f"{m.group(1)} {m.group(2)}{m.group(3)}" if m else None


def district(text):
    """Postcode district ("G74") from a full postcode, or from an address that only gives the district."""
    pc = norm_postcode(text)
    if pc:
        return pc.split()[0]
    m = re.search(OUTWARD_AT_END, str(text).upper().strip())
    return m.group(1) if m else None


def clean(v):
    """Lower case, single spaces; blank is ""."""
    return "" if pd.isna(v) else re.sub(r"\s+", " ", str(v)).strip().lower()


def text(v):
    return None if pd.isna(v) or not str(v).strip() else str(v).strip()


def weekday(v):
    """0-6 for a cell like "Monday 8th" or "tuesday", else None."""
    m = re.match(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", clean(v))
    return DAYS.index(m.group(1).title()) if m else None


def day_month(v):
    """(day, month) from text like "8th of June" or "12th - 16th October"; either can be None."""
    t = clean(v)
    d = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", t)
    m = re.search(r"\b(" + "|".join(MONTHS) + r")", t) or re.search(r"\b(" + "|".join(x[:3] for x in MONTHS) + r")\b", t)
    month = next((i + 1 for i, x in enumerate(MONTHS) if m and x.startswith(m.group(1))), None)
    return (int(d.group(1)) if d else None), month


def fmt(d):
    return f"{d.day} {MONTHS[d.month - 1].title()}"


def weeks_label(weeks):
    weeks = sorted(weeks)
    if len({w.year for w in weeks}) == 1:
        return ", ".join(fmt(w) for w in weeks) + f" {weeks[0].year}"
    return ", ".join(f"{fmt(w)} {w.year}" for w in weeks)


def week_start(day, month, wd, year, near=()):
    """(monday, guessed) for the week where `day` of `month` is weekday `wd`, or (None, False).
    Without a year, tries this year then last. Without a month, takes the match nearest `near`, else the latest."""
    today = dt.date.today()
    years = [year] if year else [today.year, today.year - 1]
    months = [month] if month else range(1, 13)
    found = []
    for y in years:
        for m in months:
            try:
                d = dt.date(y, m, day)
            except ValueError:
                continue
            if d.weekday() == wd:
                found.append(d)
    if month:
        if found:
            d = found[0]
        else:
            try:
                d = dt.date(year or today.year, month, day)
            except ValueError:
                return None, False
        return d - dt.timedelta(days=d.weekday()), False
    if not found:
        return None, False
    if near:
        d = min(found, key=lambda d: min(abs((d - n).days) for n in near))
    else:
        past = [d for d in found if d <= today + dt.timedelta(days=6)]
        d = max(past) if past else min(found)
    return d - dt.timedelta(days=d.weekday()), True


def header_rows(df, test):
    return [i for i in range(len(df)) if test([clean(v) for v in df.iloc[i]])]


def is_journey_header(cells):
    return ("day" in cells and "date" in cells and any(c.startswith("pick") for c in cells)
            and any(c.startswith("drop") for c in cells))


def is_availability_header(cells):
    has_name = any(c.startswith("volunteer") or c in ("driver", "name", "driver name") for c in cells)
    return has_name and sum(weekday(c) is not None for c in cells) >= 3


def is_mileage_header(cells):
    return "user" in cells and any("mileage" in c for c in cells)


def kind(df):
    for name, test in (("journeys", is_journey_header), ("availability", is_availability_header),
                       ("mileage", is_mileage_header)):
        if header_rows(df.head(15), test):
            return name
    return None


def journey_columns(cells):
    pick = lambda f: next((i for i, c in enumerate(cells) if f(c)), None)
    return {"day": pick(lambda c: c == "day"), "date": pick(lambda c: c == "date"),
            "department": pick(lambda c: c in ("department", "dept", "service")),
            "volunteer": pick(lambda c: c.startswith("volunteer") or c == "driver"),
            "pickup": pick(lambda c: c.startswith("pick")), "dropoff": pick(lambda c: c.startswith("drop"))}


def read_journey_tab(df, tab, year, where):
    h = header_rows(df.head(15), is_journey_header)[0]
    cols = journey_columns([clean(v) for v in df.iloc[h]])
    if cols["department"] is None:
        raise ConvertError(f"{where}: the journeys table has no DEPARTMENT column, so the service of each journey "
                           "is unknown. Add the column back with its header.")
    body = df.iloc[h + 1:]
    get = lambda r, k: r.iloc[cols[k]] if cols[k] is not None else None
    rows = [r for _, r in body.iterrows() if clean(get(r, "day")) in [d.lower() for d in DAYS]]
    if not rows:
        return None, f"Skipped {where}: it has the journeys headings but no journeys under them.", []
    # long free-text cells below the table are the coordinator's notes
    sheet_notes = [str(v).strip() for _, r in body.iterrows() if clean(get(r, "day")) not in [d.lower() for d in DAYS]
                   for v in r if isinstance(v, str) and len(v) > 40]

    first = next(((weekday(get(r, "day")), *day_month(get(r, "date"))) for r in rows
                  if day_month(get(r, "date"))[0] and day_month(get(r, "date"))[1]), None)
    if first is None:
        d, m = day_month(tab)
        first = (0, d, m) if d and m else None
    if first is None:
        raise ConvertError(f"{where}: can't tell which week this is. Write dates in the DATE column like "
                           "\"12th of October\", or name the tab like \"12th - 16th October\".")
    monday, _ = week_start(first[1], first[2], first[0], year)

    out = []
    for r in rows:
        wd = weekday(get(r, "day"))
        raw_date = get(r, "date")
        d, m = day_month(raw_date)
        if isinstance(raw_date, (dt.date, pd.Timestamp)):
            date = pd.Timestamp(raw_date).date()
        elif d and m:
            y = monday.year + (1 if m < monday.month - 6 else -1 if m > monday.month + 6 else 0)
            try:
                date = dt.date(y, m, d)
            except ValueError:
                date = monday + dt.timedelta(days=wd)
        else:
            date = monday + dt.timedelta(days=wd)
        out.append({"week_start": monday, "date": date, "day": DAYS[wd], "department": get(r, "department"),
                    "volunteer": get(r, "volunteer"), "pick_up_from": get(r, "pickup"),
                    "drop_off_to": get(r, "dropoff")})
    return pd.DataFrame(out), None, sheet_notes


def hospital_matcher(hospitals):
    by_pc = dict(zip(hospitals["postcode"], hospitals["location_id"]))

    def match(t):
        if pd.isna(t):
            return None
        pc = norm_postcode(t)
        if pc in by_pc:
            return by_pc[pc]
        return next((v for k, v in HOSPITAL_WORDS.items() if k in str(t).lower()), None)
    return match


def tidy_journeys(j, hospitals):
    hosp = hospital_matcher(hospitals)
    dept = j["department"].astype("string").str.strip().str.upper()
    j["pool"] = dept.str.startswith("CCF").map({True: "CCF", False: "GCHC"})
    j["service"] = dept.str.replace(r"^CCF\s+", "", regex=True)
    j["driver"] = j["volunteer"].astype("string").str.strip().replace("", pd.NA)
    j["driver_type"] = j["driver"].map(lambda v: None if pd.isna(v) else NON_VOLUNTEERS.get(
        v.lower(), "relay_or_partner" if "/" in v else "volunteer"))
    for side, col in (("pickup", "pick_up_from"), ("dropoff", "drop_off_to")):
        t = j[col].astype("string").str.strip()
        j[f"{side}_address"] = t.str.replace(r"\s*-\s*(pick up|collect|collection).*$", "", case=False, regex=True)
        j[f"{side}_postcode"] = t.astype(object).map(lambda v: None if pd.isna(v) else norm_postcode(v))
        j[f"{side}_hospital"] = t.astype(object).map(hosp)
        j[f"{side}_time"] = t.str.extract(TIME, flags=re.I)[0].str.replace(" ", "").str.lower()
    j["direction"] = [
        "between_hospitals" if a and b else "to_hospital" if b else "from_hospital" if a else "other"
        for a, b in zip(j["pickup_hospital"].notna(), j["dropoff_hospital"].notna())]
    # the non-hospital end: a family's home, a donor, a handover point
    j["community_postcode"] = j["pickup_postcode"].where(j["pickup_hospital"].isna(), j["dropoff_postcode"])
    community_text = j["pickup_address"].where(j["pickup_hospital"].isna(), j["dropoff_address"])
    j["community_district"] = [district(pc) if isinstance(pc, str) else None if pd.isna(t) else district(t)
                               for pc, t in zip(j["community_postcode"], community_text)]
    for c in ("week_start", "date"):
        j[c] = j[c].map(lambda d: d.isoformat())
    j["journey_id"] = [f"{w}-{i:03d}" for w, i in zip(j["week_start"], j.groupby("week_start").cumcount() + 1)]
    cols = ["journey_id", "week_start", "date", "day", "pool", "service", "driver", "driver_type",
            "pickup_address", "pickup_postcode", "pickup_hospital", "pickup_time",
            "dropoff_address", "dropoff_postcode", "dropoff_hospital", "dropoff_time",
            "direction", "community_postcode", "community_district"]
    return j[cols].reset_index(drop=True)


def read_availability_tab(df, tab, year, near, where):
    """Availability (long) and driver details from one weekly tab. Each header row starts a table; the one headed
    "Volunteer (CROSSHOUSE)" is the Crosshouse pool."""
    heads = header_rows(df, is_availability_header)
    first = [clean(v) for v in df.iloc[heads[0]]]
    days = [(i, weekday(c), day_month(c)[0]) for i, c in enumerate(first) if weekday(c) is not None]
    numbered = [(wd, d) for _, wd, d in days if d]
    month = day_month(tab)[1]
    if numbered:
        monday, guessed = week_start(numbered[0][1], month, numbered[0][0], year, near)
    else:
        d, m = day_month(tab)
        monday, guessed = week_start(d, m, 0, year) if d and m else (None, False)
    if monday is None:
        raise ConvertError(f"{where}: can't tell which week this is. Put the date in the day headings "
                           "(like \"Monday 12th\") and the month in the tab name (like \"Driver Availability - October\").")
    # a "Role" column whose cells all read like "BP - Breast Pumps" is the legend, not the drivers' roles
    is_legend = lambda i: (lambda v: len(v) and v.str.match(r"^[A-Z]+\s*-\s*\S").all())(
        df.iloc[heads[0] + 1:, i].dropna().astype(str).str.strip())
    legend_cols = [i for i, c in enumerate(first) if c in ("role", "roles") and is_legend(i)]
    legend = [str(v).strip() for i in legend_cols for v in df.iloc[heads[0] + 1:, i].dropna()]

    drivers, long, skipped = [], [], 0
    for b, h in enumerate(heads):
        cells = [clean(v) or first[i] if i < len(first) else clean(v) for i, v in enumerate(df.iloc[h])]
        name_i = next(i for i, c in enumerate(cells) if c.startswith("volunteer") or c in ("driver", "name", "driver name"))
        pick = lambda f: next((i for i, c in enumerate(cells) if f(c)), None)
        role_i = next((i for i, c in enumerate(cells) if c in ("role", "roles") and i not in legend_cols), None)
        pc_i, note_i = pick(lambda c: "postcode" in c), pick(lambda c: "note" in c)
        day_cols = [(i, weekday(c)) for i, c in enumerate(cells) if weekday(c) is not None]
        pool = "CCF" if re.search(r"crosshouse|ccf", cells[name_i]) else "GCHC"
        end = heads[b + 1] if b + 1 < len(heads) else len(df)
        for _, r in df.iloc[h + 1:end].iterrows():
            name = text(r.iloc[name_i])
            if name is None:
                continue
            filled = [text(r.iloc[i]) for i in [role_i, pc_i, *[i for i, _ in day_cols]] if i is not None]
            if len(name) > 40 or not any(filled):
                skipped += 1
                continue
            drivers.append({"driver": name, "pool": pool,
                            "roles": text(r.iloc[role_i]) if role_i is not None else None,
                            "postcode": text(r.iloc[pc_i]) if pc_i is not None else None,
                            "notes": text(r.iloc[note_i]) if note_i is not None else None})
            for i, wd in day_cols:
                raw = text(r.iloc[i])
                up = (raw or "").upper()
                cond = re.search(r"\((.*?)\)", raw or "")
                long.append({"week_start": monday.isoformat(), "date": (monday + dt.timedelta(days=wd)).isoformat(),
                             "day": DAYS[wd], "driver": name, "pool": pool,
                             "status": "available" if up.startswith("AVAILABLE") else
                                       "unavailable" if up.startswith("UNAVAILABLE") else "unknown",
                             "condition": cond.group(1) if cond else None})
    return dict(monday=monday, guessed=guessed, long=pd.DataFrame(long), drivers=pd.DataFrame(drivers),
                legend=legend, skipped=skipped, details=any(d["roles"] for d in drivers))


def latest_legend(tabs):
    return next((t["legend"] for t in sorted(tabs, key=lambda t: t["monday"], reverse=True) if t["legend"]), [])


def tidy_drivers(tabs, legend):
    """One row per driver, details from the most recent week that lists them."""
    d = pd.concat([t["drivers"].assign(_w=t["monday"]) for t in tabs if t["details"]], ignore_index=True)
    d = d.sort_values("_w", kind="stable").drop_duplicates("driver", keep="last")
    order = {n: i for i, n in enumerate(d.loc[d["_w"] == d["_w"].max(), "driver"])}
    d = d.assign(_o=d["driver"].map(order).fillna(len(order))).sort_values(["_o", "_w"], kind="stable")
    d["postcode_district"] = [district(pc) if pc else (district(raw) if raw else None)
                              for pc, raw in zip(d["postcode"].map(lambda v: norm_postcode(v) if v else None), d["postcode"])]
    d["postcode"] = d["postcode"].map(lambda v: norm_postcode(v) if v else None)
    d["roles"] = d["roles"].fillna("").str.upper().str.findall(r"[A-Z]+").str.join("|")
    codes = list(dict.fromkeys([re.split(r"\s*-\s*", l)[0].strip().upper() for l in legend]))
    codes += sorted({c for rs in d["roles"] for c in rs.split("|") if c} - set(codes))
    codes += [c for c in TRAINING if c not in codes]
    for c in codes:
        d[f"role_{c}"] = d["roles"].str.split("|").map(lambda rs, c=c: c in rs)
    d["notes"] = d["notes"].astype("string").str.strip()
    return d[["driver", "pool", "postcode", "postcode_district", "roles", *[f"role_{c}" for c in codes], "notes"]
             ].reset_index(drop=True)


def read_mileage_tab(df):
    h = header_rows(df.head(15), is_mileage_header)[0]
    cells = [clean(v) for v in df.iloc[h]]
    frames = []
    for c in [i for i, v in enumerate(cells) if v == "user"]:
        nxt = [i for i, v in enumerate(cells) if v == "user" and i > c]
        span = range(c + 1, nxt[0] if nxt else len(cells))
        rate_i = next((i for i in span if "rate" in cells[i]), None)
        miles_i = next((i for i in span if cells[i] in ("mileage", "miles")), None)
        if miles_i is None:
            continue
        label = clean(df.iloc[h - 1, c]) if h > 0 else ""
        block = pd.DataFrame({"driver": df.iloc[h + 1:, c].map(text),
                              "rate": df.iloc[h + 1:, rate_i] if rate_i is not None else None,
                              "miles": pd.to_numeric(df.iloc[h + 1:, miles_i], errors="coerce")})
        frames.append(block.assign(pool="CCF" if label in ("ccf", "crosshouse") else "GCHC"))
    if not frames:
        return None
    m = pd.concat(frames, ignore_index=True).dropna(subset=["driver", "miles"])
    rate = m["rate"].astype("string").str.extract(r"(\d+(?:\.\d+)?)\s*\)?\s*$")[0]
    m["rate_gbp_per_mile"] = pd.to_numeric(rate, errors="coerce")
    m["claim_gbp"] = (m["miles"] * m["rate_gbp_per_mile"]).round(2)
    return m[["driver", "pool", "miles", "rate_gbp_per_mile", "claim_gbp"]].reset_index(drop=True)


def hospital_table(df):
    cols = {clean(c): c for c in df.columns}
    code = next((cols[c] for c in ("location_id", "code", "hospital", "id") if c in cols), None)
    if code is None or "postcode" not in cols:
        return None
    h = pd.DataFrame({"location_id": df[code].astype(str).str.strip().str.upper(),
                      "postcode": df[cols["postcode"]].map(norm_postcode)})
    return h.dropna()


def read_csv(blob):
    """A CSV as Excel saves it (UTF-8 or Windows encoding, commas or semicolons), every cell as text, no header row
    assumed, so it can be searched for its headings like a workbook tab. None if it can't be read."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(blob), header=None, dtype=str, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return None


def from_workbooks(files, year=None):
    """files: (filename, bytes) pairs: .xlsx workbooks, or their weekly tabs saved as .csv, plus an optional
    hospital_locations .csv.
    year: the year of the weekly tabs; None tries this year then last.
    Returns (tables, notes): drivers, availability, journeys, and mileage if given; notes on what was read or skipped.
    Raises ConvertError if journeys, availability or driver details are missing."""
    tables, notes, _ = read_workbooks(files, year)
    return tables, notes


def read_workbooks(files, year=None):
    """from_workbooks, plus extras for src/clean_data.py: the role legend and the notes under the journey tables."""
    notes, skipped = [], []
    hospitals = pd.DataFrame(HOSPITALS, columns=["location_id", "name", "postcode"])[["location_id", "postcode"]]
    sheets = []  # (where, tab, kind, df)
    planner_layout = False
    for fname, blob in files:
        if str(fname).lower().endswith(".csv"):
            # a CSV is one tab saved on its own: a journeys, availability or mileage sheet, or a hospital list
            raw = read_csv(blob)
            if raw is None:
                skipped.append(f"Skipped {fname}: it couldn't be opened. Save it from Excel as 'CSV UTF-8' or as .xlsx.")
                continue
            body = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
            body.columns = range(body.shape[1])
            first = {str(c).strip().lower() for c in body.iloc[0]} if len(body) else set()
            if first & {"journey_id", "week_start", "driver_type"} or any(c.startswith("role_") for c in first):
                planner_layout = True   # the planner's own CSV layout, which goes in the other upload box
                skipped.append(f"Skipped {fname}: a CSV in the planner's layout, not one of the charity's spreadsheets.")
                continue
            if len(body) and kind(body):
                book = {Path(fname).stem: raw}
            else:
                h = hospital_table(raw.iloc[1:].set_axis(raw.iloc[0].astype(str).str.strip(), axis=1)) if len(raw) > 1 else None
                if h is not None:
                    hospitals = pd.concat([h, hospitals]).drop_duplicates("postcode")
                    notes.append(f"Hospitals: {len(h)} from {fname}, plus the built-in list.")
                else:
                    skipped.append(f"Skipped {fname}: not a journeys, availability, mileage or hospital table.")
                continue
        else:
            try:
                book = pd.read_excel(io.BytesIO(blob), sheet_name=None, header=None)
            except Exception:
                skipped.append(f"Skipped {fname}: it couldn't be opened as an Excel workbook. Save it as .xlsx and "
                               "upload it again.")
                continue
        for tab, df in book.items():
            df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
            df.columns = range(df.shape[1])
            k = kind(df) if len(df) else None
            if k:
                sheets.append((f"tab '{tab}' in {fname}", tab, k, df))
            elif len(df):
                skipped.append(f"Skipped tab '{tab}' in {fname}: not a journeys, availability or mileage table.")

    # one tab per week; on a clash keep the tab that names more drivers
    weeks = {}
    for where, tab, k, df in sheets:
        if k != "journeys":
            continue
        j, msg, sheet_notes = read_journey_tab(df, tab, year, where)
        if j is None:
            skipped.append(msg)
            continue
        w = j["week_start"].iat[0]
        score = (j["volunteer"].notna().sum(), len(j))
        if w in weeks:
            new = (where, j, score, sheet_notes)
            keep, drop = (weeks[w], new) if weeks[w][2] >= score else (new, weeks[w])
            skipped.append(f"Skipped {drop[0]}: same week as {keep[0]} ({fmt(w)}), which names more drivers.")
            weeks[w] = keep
        else:
            weeks[w] = (where, j, score, sheet_notes)

    # dated tabs first, so a tab that only says "Monday 19th" can be placed near them
    av_sheets = sorted([s for s in sheets if s[2] == "availability"], key=lambda s: day_month(s[1])[1] is None)
    av, near = {}, list(weeks)
    for where, tab, k, df in av_sheets:
        t = read_availability_tab(df, tab, year, near, where)
        t["where"] = where
        if t["long"].empty:
            skipped.append(f"Skipped {where}: it has the availability headings but no drivers under them.")
            continue
        if t["guessed"]:
            notes.append(f"Availability: {where} doesn't name its month; taken as the week of "
                         f"{fmt(t['monday'])} {t['monday'].year}.")
        if t["skipped"]:
            skipped.append(f"Skipped {t['skipped']} note rows in {where}.")
        w = t["monday"]
        if w in av:
            skipped.append(f"Skipped {where}: same week as {av[w]['where']} ({fmt(w)}).")
            continue
        av[w] = t
        near.append(w)

    mileage = None
    for where, tab, k, df in sheets:
        if k == "mileage" and mileage is None:
            mileage = read_mileage_tab(df)
            if mileage is None or mileage.empty:
                skipped.append(f"Skipped {where}: no drivers with miles under the User / Mileage headings.")
                mileage = None

    missing = []
    if not weeks:
        missing.append("the journeys workbook (weekly tabs headed DAY, DATE, DEPARTMENT, VOLUNTEER, PICK UP FROM, "
                       "DROP OFF TO)")
    if not av:
        missing.append("the availability workbook (weekly tabs headed Role, Volunteer, Driver Postcode, then a "
                       "column per day such as \"Monday 12th\")")
    elif not any(t["details"] for t in av.values()):
        missing.append("the driver details (an availability tab with each driver's Role and Driver Postcode "
                       "columns, as in the Drivers Info workbook)")
    if missing:
        looked = "; ".join(skipped) if not sheets and skipped else " ".join(s for s in skipped if "couldn't be opened" in s)
        if planner_layout:   # the planner's own CSVs, added in the box for the charity's spreadsheets
            raise ConvertError("These look like CSV files in the planner's layout (drivers.csv, availability.csv, "
                               "journeys.csv). Choose \"Three CSV files in the planner's layout\" in the panel on the "
                               "left and add them there.")
        raise ConvertError("Couldn't find " + " or ".join(missing) + ". Upload the workbooks as the charity keeps them "
                           "(.xlsx, or each weekly tab saved as CSV), with their header rows." + (f" ({looked})" if looked else ""))

    jw = sorted(weeks)
    journeys = tidy_journeys(pd.concat([weeks[w][1] for w in jw], ignore_index=True), hospitals)
    tabs = [av[w] for w in sorted(av)]
    availability = pd.concat([t["long"] for t in tabs], ignore_index=True)
    legend = latest_legend(tabs)
    drivers = tidy_drivers(tabs, legend)
    tables = {"drivers": drivers, "availability": availability, "journeys": journeys}

    notes.insert(0, f"Journeys: {len(jw)} weekly {'tab' if len(jw) == 1 else 'tabs'}, {len(journeys)} journeys "
                    f"({weeks_label(jw)}).")
    notes.insert(1, f"Availability: {len(av)} weekly {'tab' if len(av) == 1 else 'tabs'}, "
                    f"{availability['driver'].nunique()} drivers ({weeks_label(av)}).")
    latest = max(w for w, t in av.items() if t["details"])
    notes.insert(2, f"Driver details: {len(drivers)} drivers, from {av[latest]['where']} (week of {fmt(latest)} "
                    f"{latest.year}).")
    if mileage is not None:
        tables["mileage"] = mileage
        notes.append(f"Mileage: {len(mileage)} drivers, {mileage['miles'].sum():,.0f} miles.")
    # round-trip through CSV so the types match an uploaded CSV
    tables = {k: pd.read_csv(io.StringIO(df.to_csv(index=False))) for k, df in tables.items()}
    sheet_notes = list(dict.fromkeys(n for w in jw for n in weeks[w][3]))
    return tables, notes + skipped, {"legend": legend, "sheet_notes": sheet_notes}
