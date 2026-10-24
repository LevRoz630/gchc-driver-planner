"""What the three upload files must look like, and checks that explain problems in plain English.

Problems that stop the planner are errors; things it can work around (a postcode it doesn't know, a blank
service tag) are warnings, shown with the rows affected so the coordinator can fix the spreadsheet."""
import re

import numpy as np
import pandas as pd

import planner
from convert import NON_VOLUNTEERS

STATUSES = {"available", "unavailable", "unknown"}
DRIVER_TYPES = {"volunteer", "relay_or_partner", "taxi"}
SERVICES = planner.SERVICES | {"NICU"}   # NICU is read as the neonatal unit

# For each file: (column, what goes in it, example). Examples are made up.
FORMAT = {
    "drivers": [
        ("driver", "The driver's name, written the same way in all three files", "Driver A"),
        ("postcode", "Where the driver lives", "G2 1DU"),
        ("role_PAT", "Yes if trained for patient transport, otherwise No (TRUE/FALSE also works)", "Yes"),
        ("role_RESP", "Respiratory", "FALSE"),
        ("role_MB", "Milk bank", "TRUE"),
        ("role_BP", "Breast pumps", "TRUE"),
        ("role_NNU", "Neonatal unit", "FALSE"),
    ],
    "availability": [
        ("driver", "The driver's name, as in the drivers list", "Driver A"),
        ("week_start", "The Monday of that week (optional: worked out from the date if left out)", "2026-10-12"),
        ("date", "The day, e.g. 13/10/2026 or 2026-10-13", "2026-10-13"),
        ("status", "available, unavailable, or blank if not known", "available"),
    ],
    "journeys": [
        ("week_start", "The Monday of that week (optional: worked out from the date if left out)", "2026-10-12"),
        ("date", "The day of the journey, e.g. 13/10/2026 or 2026-10-13", "2026-10-13"),
        ("service", "PAT, RESP, MB, BP, NNU, NICU or MM (or the name, like Patient transport)", "PAT"),
        ("driver", "Who drove it, as in the drivers list (blank if not recorded)", "Driver A"),
        ("driver_type", "volunteer, relay_or_partner or taxi (blank means volunteer if a driver is named)", "volunteer"),
        ("pickup_postcode", "Where the journey starts", "G51 4TF"),
        ("pickup_hospital", "A short hospital name if the journey starts at a hospital, otherwise blank", "RHC"),
        ("dropoff_postcode", "Where the journey ends", "PA4 8NT"),
        ("dropoff_hospital", "A short hospital name if the journey ends at a hospital, otherwise blank", ""),
    ],
    # optional fourth file: the yearly mileage report
    "mileage": [
        ("driver", "The driver's name, as in the drivers list", "Driver A"),
        ("miles", "Miles claimed in the year", "1920"),
        ("claim_gbp", "Amount paid in pounds (optional)", "864.00"),
        ("rate_gbp_per_mile", "Pence per mile as pounds, if there's no claim column (optional)", "0.45"),
    ],
}
OPTIONAL = {"dropoff_hospital", "claim_gbp", "rate_gbp_per_mile", "week_start", "driver_type"}
REQUIRED = {f: [c for c, _, _ in cols if not c.startswith("role_") and c not in OPTIONAL]
            for f, cols in FORMAT.items()}


YES = {"true", "yes", "y", "x", "1", "1.0", "✓", "✔", "trained"}
SERVICE_WORDS = {"patient transport": "PAT", "patient": "PAT", "respiratory": "RESP", "resp": "RESP",
                 "milk bank": "MB", "milk": "MB", "breast pump": "BP", "neonatal intensive": "NICU", "nicu": "NICU",
                 "neonatal": "NNU", "memory making": "MM", "cardio": "CARDIO", "biopic": "BIO"}
# other headings people use for the same column
ALIASES = {"name": "driver", "volunteer": "driver", "post_code": "postcode", "date_of_journey": "date",
           "department": "service", "availability": "status", "week": "week_start", "week_commencing": "week_start",
           "type": "driver_type"}


def key(heading):
    """'Post Code ' -> 'post_code': lower case, spaces and dashes as underscores."""
    return re.sub(r"[\s\-]+", "_", str(heading).strip().lower()).strip("_")


def dates(col):
    """Dates written as 13/10/2026, 2026-10-13, 13 Oct 2026, with a time, or as Excel date numbers -> 2026-10-13
    (blank where it isn't a date)."""
    text = col.astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.to_datetime(text.where(iso), errors="coerce", format="mixed")
    out = out.fillna(pd.to_datetime(text.where(~iso), errors="coerce", dayfirst=True, format="mixed"))
    serial = pd.to_numeric(col, errors="coerce")
    excel = serial.between(20000, 80000)
    out = out.where(~excel, pd.Timestamp("1899-12-30") + pd.to_timedelta(serial.where(excel), unit="D"))
    return out.dt.strftime("%Y-%m-%d").where(out.notna())


def service_code(v):
    if not isinstance(v, str) or not v.strip():
        return v
    t = v.strip()
    if t.upper() in SERVICES:
        return t.upper()
    words = re.findall(r"[A-Za-z]+", t.upper())
    codes = [w for w in words if w in SERVICES]
    if codes:                       # "CCF RESP" -> RESP
        return codes[-1]
    low = t.lower()
    return next((code for word, code in SERVICE_WORDS.items() if word in low), t)


def tidy(name, df):
    """Forgive the usual spreadsheet differences before checking: headings in any case or spacing, blank rows,
    stray spaces, UK dates, Yes/No for training, blank status, service names written out. Returns the tidied
    table and a list of plain-English notes on what was changed."""
    notes = []
    df = df.dropna(how="all").copy()
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") and df[c].isna().all()]   # Excel's empty columns
    df = df.drop(columns=unnamed)
    known = {c for cols in FORMAT.values() for c, _, _ in cols} | {"notes", "condition", "pickup_time", "dropoff_time"}
    rename = {}
    for c in df.columns:
        k = key(c)
        k = ALIASES.get(k, k)
        m = re.fullmatch(r"role_(\w+)", k)
        if m:
            k = f"role_{m.group(1).upper()}"
        if k != c and (k in known or k.startswith("role_")) and k not in df.columns and k not in rename.values():
            rename[c] = k
    df = df.rename(columns=rename)
    if rename:
        notes.append(f"Read the {name} headings " + ", ".join(f"'{a}' as {b}" for a, b in rename.items()) + ".")
    if name == "drivers" and "roles" in df.columns and not any(c.startswith("role_") for c in df.columns):
        for code in sorted(SERVICES - {"NICU"}):     # one "roles" column like "RESP|BP|PAT" -> role_ columns
            df[f"role_{code}"] = df["roles"].astype(str).str.upper().str.contains(rf"\b{code}\b")
        notes.append("Split the drivers' roles column into one yes/no column per service.")
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v).replace("", np.nan)
    for c in ("date", "week_start"):
        if c in df.columns:
            df[c] = dates(df[c])
    if "date" in df.columns and name in ("availability", "journeys"):
        d = pd.to_datetime(df["date"], errors="coerce")
        df["week_start"] = (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d").where(d.notna())
    for c in [c for c in df.columns if c.startswith("role_")]:
        df[c] = df[c].map(lambda v: str(v).strip().lower() in YES if pd.notna(v) else False)
    if "status" in df.columns:
        st = df["status"].astype(str).str.strip().str.lower()
        st = st.replace({"nan": "unknown", "free": "available", "yes": "available", "y": "available",
                         "no": "unavailable", "n": "unavailable", "not available": "unavailable", "?": "unknown"})
        df["status"] = st.where(~st.str.startswith("available"), "available")
    if "service" in df.columns:
        df["service"] = df["service"].map(service_code)
    if name == "journeys" and "driver" in df.columns:
        t = df["driver_type"].astype(str).str.strip().str.lower() if "driver_type" in df.columns else pd.Series("nan", index=df.index)
        t = t.str.replace(" ", "_").replace({"relay": "relay_or_partner", "partner": "relay_or_partner"})
        name_col = df["driver"].astype(str)
        known = name_col.str.strip().str.lower().map(NON_VOLUNTEERS)
        guess = known.fillna(pd.Series(np.where(name_col.str.lower().str.contains("taxi"), "taxi",
                                                np.where(name_col.str.contains("/"), "relay_or_partner", "volunteer")),
                                       index=df.index))
        df["driver_type"] = t.where(t != "nan", guess.where(df["driver"].notna()))
    for c in ("miles", "claim_gbp", "rate_gbp_per_mile"):     # "£864.41", "1,920", "Cars and Vans (0.45)"
        if c in df.columns:
            num = df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+(?:\.\d+)?)")[0]
            df[c] = pd.to_numeric(num, errors="coerce")
    for c in ("pickup_hospital", "dropoff_hospital"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: v.strip().upper() if isinstance(v, str) else v)
    return df, notes


def n(count, one, many):
    """'1 journey has' / '3 journeys have'."""
    return f"{count} {one if count == 1 else many}"


def template(name):
    """A one-row example file to download and fill in."""
    return pd.DataFrame([{c: ex for c, _, ex in FORMAT[name]}]).to_csv(index=False).encode()


def guide(name):
    return pd.DataFrame([{"Column": c, "What goes in it": what, "Example": ex} for c, what, ex in FORMAT[name]])


def rows(df, cols=()):
    """The rows concerned, by their row number in the spreadsheet (header is row 1), without postcodes or
    addresses so the list is safe to show on screen."""
    return df[list(cols)].assign(**{"Row in file": df.index + 2})[["Row in file", *cols]]


def which_file(df):
    """Guess which of the three files a table is, from its columns."""
    best = max(REQUIRED, key=lambda f: len(set(REQUIRED[f]) & set(df.columns)))
    return best if set(REQUIRED[best]) <= set(df.columns) else None


def validate_mileage(mileage):
    """Problems with the optional mileage file, as a list of (message, optional DataFrame of rows)."""
    if mileage.empty:
        return [("The mileage file has no rows.", None)]
    missing = [c for c in REQUIRED["mileage"] if c not in mileage.columns]
    if missing:
        looks_like = which_file(mileage)
        hint = (f" It looks like the {looks_like} file, so it may be in the wrong box." if looks_like and
                looks_like != "mileage" else " See 'What the files should look like' for the column names.")
        return [(f"The mileage file is missing these columns: {', '.join(missing)}.{hint}", None)]
    bad = mileage[pd.to_numeric(mileage["miles"], errors="coerce").isna()]
    if len(bad):
        return [(f"{n(len(bad), 'row', 'rows')} in the mileage file {'has' if len(bad) == 1 else 'have'} miles that "
                 "aren't a number.", rows(bad, ["driver"]).head(10))]
    return []


def validate(tables, coords):
    """tables: {"drivers": df, "availability": df, "journeys": df}. Returns (errors, warnings), each a list of
    (message, optional DataFrame of the rows concerned)."""
    errors, warnings = [], []
    for name, df in tables.items():
        if df.empty:
            errors.append((f"The {name} file has no rows.", None))
            continue
        missing = [c for c in REQUIRED[name] if c not in df.columns]
        if missing:
            looks_like = which_file(df)
            hint = (f" It looks like the {looks_like} file, so it may be in the wrong box." if looks_like and
                    looks_like != name else " See 'What the files should look like' for the column names.")
            errors.append((f"The {name} file is missing these columns: {', '.join(missing)}.{hint}", None))
    if errors:
        return errors, warnings
    drivers, availability, journeys = tables["drivers"], tables["availability"], tables["journeys"]

    if not any(c.startswith("role_") for c in drivers.columns):
        errors.append(("The drivers file has no training columns (role_PAT, role_MB and so on), so the planner "
                       "can't tell who is trained for what.", None))
    for name, df in (("availability", availability), ("journeys", journeys)):
        bad = df[df["date"].isna()]
        if len(bad):
            errors.append((f"{n(len(bad), 'row', 'rows')} in the {name} file {'has' if len(bad) == 1 else 'have'} a date the "
                           "planner can't read. Write dates like 13/10/2026.",
                           rows(bad, ["driver"] if name == "availability" else ["service"]).head(10)))
    if errors:
        return errors, warnings

    bad = availability[~availability["status"].astype(str).str.strip().str.lower().isin(STATUSES)]
    if len(bad):
        warnings.append((f"{n(len(bad), 'availability row has', 'availability rows have')} a status other than available, unavailable or "
                         "unknown. They are treated as not available.", rows(bad, ["driver", "date", "status"])))
    bad = journeys[~journeys["service"].isin(SERVICES)]
    if len(bad):
        warnings.append((f"{n(len(bad), 'journey has', 'journeys have')} a service the planner doesn't know "
                         f"({', '.join(map(str, bad['service'].unique()[:5]))}). They are left out.", rows(bad, ["date", "service"])))
    bad = journeys[journeys["driver_type"].notna() & ~journeys["driver_type"].isin(DRIVER_TYPES)]
    if len(bad):
        warnings.append((f"{n(len(bad), 'journey has', 'journeys have')} a driver_type other than volunteer, relay_or_partner or taxi.",
                         rows(bad, ["date", "driver_type"])))

    named = journeys.groupby("week_start")["driver"].apply(lambda d: d.notna().mean())
    if not (named >= 0.5).any():
        errors.append(("No week in the journeys file says who drove most of its journeys. Fill in the driver "
                       "column for at least one full week.", None))
        return errors, warnings
    weeks = set(named[named >= 0.5].index)
    when = ", ".join(pd.to_datetime(sorted(weeks)).strftime("%-d %B %Y")[-3:])
    if journeys[["pickup_hospital"] + [c for c in ["dropoff_hospital"] if c in journeys]].isna().all().all():
        errors.append(("No journey says which hospital it starts or ends at. Put a short hospital name (like RHC) "
                       "in pickup_hospital or dropoff_hospital.", None))
    pcs = set(coords.index)
    known = lambda s: s.astype(str).str.replace(r"\s+", "", regex=True).str.upper().map(pcs.__contains__)
    legs = journeys[journeys["week_start"].isin(weeks) & (journeys["driver_type"] == "volunteer")
                    & journeys["driver"].isin(set(drivers["driver"]))]
    if legs.empty:
        errors.append((f"In the weeks that record who drove (week of {when}), no journey has a volunteer whose name "
                       "is in the drivers list. Check the names are spelled the same in both files.", None))
    elif not (known(legs["pickup_postcode"]) & known(legs["dropoff_postcode"])).any():
        errors.append((f"In the weeks that record who drove (week of {when}), no volunteer journey has a pickup and "
                       "dropoff postcode the planner recognises, so it can't work out distances.", None))
    if not availability["week_start"].isin(weeks).any():
        errors.append((f"The availability file has no days in the week that records who drove (week of {when}). Add "
                       "that week's availability.", None))
    drv_known = known(drivers["postcode"])
    if not drv_known.any():
        errors.append(("None of the drivers' postcodes is a current Scottish postcode, so the planner can't place "
                       "anyone on the map.", None))
    if errors:
        return errors, warnings
    stray = sorted(set(availability["driver"].dropna()) - set(drivers["driver"]))
    if stray:
        warnings.append((f"{n(len(stray), 'name in the availability file isn\'t', 'names in the availability file aren\'t')} in the drivers "
                         "list, so the planner never sees them as free. Check the spelling matches.",
                         pd.DataFrame({"driver": stray})))
    volunteer = journeys[journeys["driver_type"] == "volunteer"]
    unknown = sorted(set(volunteer["driver"].dropna()) - set(drivers["driver"]))
    if unknown:
        warnings.append((f"{n(len(unknown), 'volunteer name in the journeys file isn\'t', 'volunteer names in the journeys file aren\'t')} in the drivers list, so "
                         "those journeys are left out. Check the names are spelled the same, with one name per "
                         "journey.",
                         pd.DataFrame({"driver": unknown})))
    dup = drivers[drivers["driver"].duplicated(keep=False)]
    if len(dup):
        warnings.append(("Some drivers appear twice in the drivers list. The first row is used.", rows(dup, ["driver"])))

    blank = drivers["postcode"].isna()
    bad = drivers[~blank & ~known(drivers["postcode"])]
    if blank.any():
        warnings.append((f"{n(int(blank.sum()), 'driver has', 'drivers have')} no postcode, so they can't be matched to nearby journeys.",
                         rows(drivers[blank], ["driver"])))
    if len(bad):
        warnings.append((f"{n(len(bad), 'driver has', 'drivers have')} a postcode that isn't a current Scottish postcode (it may be "
                         "mistyped or out of date), so they can't be matched to nearby journeys.",
                         rows(bad, ["driver"])))
    for col in ("pickup_postcode", "dropoff_postcode"):
        where = col.split("_")[0]
        blank = journeys[col].isna()
        bad = journeys[~blank & ~known(journeys[col])]
        if blank.any():
            warnings.append((f"{n(int(blank.sum()), 'journey has', 'journeys have')} no {where} postcode. They still count as journeys but "
                             "are left out of distances and costs.", rows(journeys[blank], ["date", "service"])))
        if len(bad):
            warnings.append((f"{n(len(bad), 'journey has', 'journeys have')} a {where} postcode that isn't a current Scottish postcode "
                             "(it may be mistyped or out of date). They are left out of distances and costs.",
                             rows(bad, ["date", "service"])))
    return errors, warnings
