"""Who could do this journey? Suggested drivers for one journey, or for every journey that has no driver yet.

The same order as docs/matching.md: trained for the service, free that day (respecting the conditions written on
the availability sheet), within reach of the pickup, then fewest journeys recently and nearest. The coordinator
still chooses: each driver's own notes are shown next to them."""
import re

import pandas as pd

import streamlit as st

import ui

from planner import HOSPITAL_NAMES, MILES_PER_KM, SERVICE_NAMES, hospitals, km

TIMES = ["Any time", "Morning", "Afternoon or evening"]
RECENT_DAYS = 28    # "journeys recently" = in the four weeks up to the day


def clock(t):
    """'8.30am', '8:30am', '2pm' -> 'Morning' / 'Afternoon or evening'; anything else -> 'Any time'."""
    m = re.match(r"\s*(\d{1,2})(?:[.:](\d{2}))?\s*([ap]m)", str(t).lower())
    if not m:
        return "Any time"
    return "Morning" if m.group(3) == "am" and int(m.group(1)) != 12 else "Afternoon or evening"


def condition_allows(condition, when, days_that_week):
    """Whether a condition written on the availability sheet allows this journey, and why not if it doesn't."""
    c = str(condition).strip().lower() if isinstance(condition, str) else ""
    if not c:
        return True, ""
    if c.startswith("no am") and when == "Morning":
        return False, "no morning journeys that day"
    if c in ("am", "am only") and when == "Afternoon or evening":
        return False, "mornings only that day"
    if c.startswith("one day") and days_that_week > 0:
        return False, "only one day a week, and already has one"
    return True, ""


def suggest(net, service, date, when, lat, lon):
    """(suggested, worth_asking): drivers trained for `service` and within reach of (lat, lon). Suggested ones are
    marked free that day and their conditions allow it; worth asking ones left that day blank."""
    drv = net.drv[net.drv["lat"].notna()]
    trained = drv.get(f"role_{service}", pd.Series(False, index=drv.index)).astype(bool)
    dist = pd.Series(km(drv["lat"].values, drv["lon"].values, lat, lon, net.s.road_factor), index=drv.index)
    pool = drv[trained & (dist <= net.pickup_reach_km)]

    av = net.availability[net.availability["date"].astype(str) == str(date)]
    status = av.assign(s=av["status"].astype(str).str.strip().str.lower()).drop_duplicates("driver").set_index("driver")
    j = net.journeys[net.journeys["driver"].notna()]
    day = pd.to_datetime(date)
    jd = pd.to_datetime(j["date"])
    recent = j[(jd <= day) & (jd > day - pd.Timedelta(days=RECENT_DAYS))]["driver"].value_counts()
    week = j[(jd >= day - pd.Timedelta(days=day.weekday())) & (jd < day - pd.Timedelta(days=day.weekday() - 7))
             & (jd != day)].groupby("driver")["date"].nunique()
    today = j[jd == day]["driver"].value_counts()

    rows, ask = [], []
    for name in pool.index:
        s = status["s"].get(name, "unknown")
        cond = status["condition"].get(name) if "condition" in status else None
        base = {"driver": name, "district": pool.at[name, "postcode_district"] if "postcode_district" in pool else "",
                "miles": dist[name] * MILES_PER_KM, "recent": int(recent.get(name, 0)),
                "notes": pool.at[name, "notes"] if "notes" in pool and isinstance(pool.at[name, "notes"], str) else ""}
        if s == "available":
            ok, why = condition_allows(cond, when, int(week.get(name, 0)))
            if ok:
                that_day = [str(cond)] if isinstance(cond, str) and cond.strip() else []
                if today.get(name, 0):
                    that_day.append(f"already has {today[name]} journey{'s' if today[name] > 1 else ''} that day")
                rows.append({**base, "that_day": "; ".join(that_day) or "free"})
        elif s not in ("unavailable",):
            ask.append(base)
    order = ["recent", "miles"]
    suggested = pd.DataFrame(rows, columns=["driver", "district", "miles", "recent", "notes", "that_day"])
    asking = pd.DataFrame(ask, columns=["driver", "district", "miles", "recent", "notes"])
    return suggested.sort_values(order), asking.sort_values(order)


def table(df, extra=None):
    out = pd.DataFrame({"Driver": df["driver"], "Area": df["district"],
                        "Distance (miles)": df["miles"].round(0).astype(int),
                        f"Journeys in the last {RECENT_DAYS // 7} weeks": df["recent"]})
    if extra:
        out[extra[0]] = df[extra[1]]
    out["Their notes"] = df["notes"]
    return out


def render(net, centres):
    st.markdown("Pick a journey to see which drivers could take it: trained for the service, marked free that day, "
                "and living within reach of the pickup. Drivers with the fewest journeys recently come first, so "
                "the work is shared out. Check their notes before asking.")
    unfilled = no_driver_yet(net)
    modes = ["One journey"] + (["Journeys without a driver yet"] if len(unfilled) else [])
    mode = st.radio("Show", modes, horizontal=True) if len(modes) > 1 else modes[0]

    if mode == "One journey":
        services = [s for s in SERVICE_NAMES if f"role_{s}" in net.drv.columns and net.drv[f"role_{s}"].any()]
        c = st.columns(3)
        service = c[0].selectbox("Service", services, format_func=lambda s: SERVICE_NAMES.get(s, s))
        dates = sorted(net.availability["date"].astype(str).unique(), reverse=True)
        date = c[1].selectbox("Day", dates, format_func=lambda d: pd.to_datetime(d).strftime("%A %-d %B %Y"))
        when = c[2].radio("Time", TIMES, horizontal=True)
        places = hospitals_in(net)
        start = st.selectbox("Starts at", [*places.index, "Somewhere else (type the postcode area)"])
        if start in places.index:
            lat, lon = places.at[start, "lat"], places.at[start, "lon"]
        else:
            area = st.text_input("Postcode area, like G33 or KA1", "").strip().upper().replace(" ", "")
            if area not in centres.index:
                if area:
                    st.warning(f"{area} isn't a postcode area we know. Type only the first part of the postcode.")
                return
            lat, lon = centres.at[area, "lat"], centres.at[area, "lon"]
        show_one(net, service, date, when, lat, lon)
    else:
        show_unfilled(net, unfilled)

    with st.expander("How this is worked out"):
        st.markdown(f"""
A driver is suggested if they are trained for the service, marked free that day on the availability sheet, and
live within {net.pickup_reach_km * MILES_PER_KM:.0f} miles of the pickup by road (most of today's journeys start
that close to the driver's home). Conditions written on the availability sheet are respected: "AM" means mornings
only, "No AM journeys" means no mornings, and "one day" means one day a week. Drivers already driving that day
are still listed, because two journeys can often be done in one day.

They are listed with the fewest journeys in the last {RECENT_DAYS // 7} weeks first, then the nearest. Drivers who
left that day blank are listed separately as worth asking. The planner can't read drivers' notes, so it shows them:
check them before asking.
""")


def show_one(net, service, date, when, lat, lon):
    suggested, asking = suggest(net, service, date, when, lat, lon)
    name = SERVICE_NAMES.get(service, service).lower()
    st.header("Drivers who could take it")
    if suggested.empty:
        st.warning(f"Nobody trained for {name}, living within reach, is marked free then.")
    else:
        ui.table(table(suggested, ("That day", "that_day")), key="suggested")
    if len(asking):
        st.header("Worth asking: left that day blank")
        st.markdown(f"Trained for {name} and living within reach, but they haven't said whether they're free.")
        ui.table(table(asking), key="asking")


def no_driver_yet(net):
    """Journeys after the latest week that records who drove, with no driver and no taxi or partner yet: the week
    being planned."""
    j = net.journeys
    later = pd.to_datetime(j["week_start"]) > pd.to_datetime(max(net.driver_weeks))
    return j[later & j["driver"].isna() & j["driver_type"].isna()]


def show_unfilled(net, unfilled):
    st.header("Journeys without a driver yet")
    st.markdown("The three best-placed drivers for each journey. The same driver can appear for several journeys; "
                "check the list from the top so nobody is asked twice for the same day.")
    js = net.place(unfilled, "pickup_postcode")
    out = []
    for r in js.itertuples():
        if pd.isna(r.lat):
            names = "Pickup postcode not recognised"
        else:
            s, _ = suggest(net, r.service, r.date, clock(getattr(r, "pickup_time", "")), r.lat, r.lon)
            names = ", ".join(f"{d.driver} ({d.miles:.0f} mi)" for d in s.head(3).itertuples()) or "Nobody free: ask the blanks"
        out.append({"Day": pd.to_datetime(r.date).strftime("%a %-d %b"), "Service": SERVICE_NAMES.get(r.service, r.service),
                    "From": r.pickup_hospital if isinstance(r.pickup_hospital, str) else "home",
                    "To": r.dropoff_hospital if isinstance(r.dropoff_hospital, str) else "home",
                    "Suggested drivers": names})
    ui.table(pd.DataFrame(out), key="unfilled")


def hospitals_in(net):
    """Hospitals the journeys start or end at, by name, busiest first, with where they are."""
    return hospitals(net).set_index("name")[["lat", "lon"]]
