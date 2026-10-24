"""Draft next week's rota: a suggested driver for every journey in a week that has no drivers yet.

The rules are the same as "Who could do this journey?" (match.py) and docs/matching.md: trained for the service,
marked free that day (respecting the conditions on the availability sheet) and within reach of the pickup. Then,
for the whole week at once:

- the hardest journeys first (the ones with the fewest suitable drivers);
- a driver already driving from the same hospital that day first, then one already driving that day, so journeys
  are grouped into driver-days the way coordinators do it;
- then whoever has the fewest days so far that week, so the work is shared out; then the nearest;
- nobody gets more journeys in a day than three in four driver-days had in the records (a typical busy day).

Every pick says why, so a coordinator can follow and change it. A draft to edit, not a booking."""
from collections import defaultdict

import pandas as pd

import streamlit as st

import ui
from match import clock, condition_allows, suggest
from planner import HOSPITAL_NAMES, MILES_PER_KM, SERVICE_NAMES, km


def weeks_to_plan(net):
    """Weeks in the journeys file where most journeys have no driver yet, latest first."""
    j = net.journeys
    named = j.groupby("week_start")["driver"].apply(lambda d: d.notna().mean())
    return sorted(named[named < 0.5].index, reverse=True)


def area(postcode):
    return str(postcode).replace(" ", "").upper()[:-3] if isinstance(postcode, str) and len(postcode) > 4 else "?"


def end(hospital, postcode):
    return HOSPITAL_NAMES.get(hospital, hospital) if isinstance(hospital, str) else f"home ({area(postcode)})"


def draft(net, week):
    """(rota, unfilled): one row per journey that the draft fills, and one per journey it can't."""
    j = net.journeys
    todo = j[(j["week_start"] == week) & j["driver"].isna()
             & ~j["driver_type"].isin(["taxi", "relay_or_partner"])]
    todo = net.place(todo, "pickup_postcode")
    per_day = net.legs.groupby(["driver", "date"]).size()
    cap = max(int(per_day.quantile(0.75)), 1) if len(per_day) else 3
    placed = net.drv[net.drv["lat"].notna()]
    av = net.availability
    cond = {(d, str(t)): c for d, t, c in zip(av["driver"], av["date"], av.get("condition", pd.Series(index=av.index)))
            if isinstance(c, str) and c.strip()}

    options, ask = {}, {}
    for i, r in todo.iterrows():
        if pd.isna(r["lat"]):
            options[i], ask[i] = pd.DataFrame(columns=["driver", "miles", "notes", "district"]), []
            continue
        s, a = suggest(net, r["service"], r["date"], clock(r.get("pickup_time")), r["lat"], r["lon"])
        options[i], ask[i] = s, list(a["driver"].head(3))

    on_day = defaultdict(int)           # (driver, date) -> journeys given
    hosp_on_day = defaultdict(set)      # (driver, date) -> hospitals they're driving from or to
    days = defaultdict(set)             # driver -> dates given this week
    filled, missing = [], []
    hospital_of = lambda r: r["pickup_hospital"] if isinstance(r["pickup_hospital"], str) else r["dropoff_hospital"]
    for i in sorted(options, key=lambda i: (len(options[i]), todo.at[i, "date"])):
        r, s, date = todo.loc[i], options[i], str(todo.at[i, "date"])
        when, hosp = clock(r.get("pickup_time")), hospital_of(r)
        free = []
        for c in s.itertuples():
            if on_day[(c.driver, date)] >= cap:
                continue
            ok, _ = condition_allows(cond.get((c.driver, date)), when, len(days[c.driver] - {date}))
            if not ok:
                continue
            group = 0 if hosp in hosp_on_day[(c.driver, date)] else (1 if on_day[(c.driver, date)] else 2)
            free.append((group, len(days[c.driver] - {date}), c.miles, c))
        row = {"date": date, "time": r.get("pickup_time") if isinstance(r.get("pickup_time"), str) else "",
               "service": r["service"], "pickup_hospital": r["pickup_hospital"], "pickup_postcode": r["pickup_postcode"],
               "dropoff_hospital": r["dropoff_hospital"], "dropoff_postcode": r["dropoff_postcode"],
               "pickup_address": r.get("pickup_address"), "dropoff_address": r.get("dropoff_address")}
        if not free:
            trained = placed.get(f"role_{r['service']}", pd.Series(False, index=placed.index)).astype(bool)
            near = (km(placed["lat"].values, placed["lon"].values, r["lat"], r["lon"], net.s.road_factor)
                    <= net.pickup_reach_km) if pd.notna(r["lat"]) else trained & False
            if pd.isna(r["lat"]):
                why = "the pickup postcode isn't recognised"
            elif not (trained & near).any():
                why = "no trained driver lives within reach of the pickup (long runs usually go to a partner charity)"
            elif s.empty:
                why = "nobody trained for it, living within reach, is marked free that day"
            else:
                why = f"every suitable driver already has {cap} journeys that day, or their conditions rule it out"
            missing.append({**row, "why": why, "ask": ", ".join(ask[i])})
            continue
        group, n_days, miles, c = min(free, key=lambda f: f[:3])
        if len(s) == 1:
            why = "the only suitable driver marked free"
        elif group == 0:
            why = "already driving from or to the same hospital that day"
        elif group == 1:
            why = "already driving that day"
        else:
            why = "fewest days this week" + (", and the nearest" if miles <= min(f[2] for f in free) else "")
        on_day[(c.driver, date)] += 1
        hosp_on_day[(c.driver, date)].add(hosp)
        days[c.driver].add(date)
        filled.append({**row, "driver": c.driver, "district": c.district, "miles": c.miles, "why": why,
                       "notes": c.notes, "choices": len(s)})
    cols = ["date", "time", "service", "pickup_hospital", "pickup_postcode", "dropoff_hospital", "dropoff_postcode",
            "pickup_address", "dropoff_address"]
    rota = pd.DataFrame(filled, columns=[*cols, "driver", "district", "miles", "why", "notes", "choices"])
    unfilled = pd.DataFrame(missing, columns=[*cols, "why", "ask"])
    return rota.sort_values(["date", "driver"]), unfilled.sort_values("date"), cap


def day(d):
    return pd.to_datetime(d).strftime("%a %-d %b")


def render(net):
    st.markdown("A first draft of who drives each journey in a week that has no drivers yet, matched the way "
                "coordinators do it: trained, free that day and nearby, with journeys on the same day grouped "
                "together and the work shared out. Every suggestion says why. It's a draft to check and change, "
                "not a booking: read each driver's notes before asking them.")
    weeks = weeks_to_plan(net)
    if not weeks:
        st.info("Every week in the journeys file already has its drivers. When next week's journeys come in, add "
                "them with the driver column left blank and they'll be drafted here.")
        return
    week = st.selectbox("Week to draft", weeks, format_func=lambda w: pd.to_datetime(w).strftime("Week of %-d %B %Y"))
    with st.spinner("Drafting the week..."):
        rota, unfilled, cap = draft(net, week)
    total = len(rota) + len(unfilled)
    if total == 0:
        st.info("That week has no journeys waiting for a volunteer.")
        return

    c = st.columns(4)
    c[0].metric("Journeys to fill", total)
    c[1].metric("Filled by the draft", f"{len(rota)} ({len(rota) / total:.0%})")
    c[2].metric("Drivers used", rota["driver"].nunique())
    busiest = rota.groupby("driver")["date"].nunique()
    c[3].metric("Most days for one driver", int(busiest.max()) if len(busiest) else 0)

    st.header("The draft")
    ui.table(pd.DataFrame({
        "Day": rota["date"].map(day), "Time": rota["time"],
        "Service": rota["service"].map(lambda s: SERVICE_NAMES.get(s, s)),
        "From": [end(h, p) for h, p in zip(rota["pickup_hospital"], rota["pickup_postcode"])],
        "To": [end(h, p) for h, p in zip(rota["dropoff_hospital"], rota["dropoff_postcode"])],
        "Suggested driver": rota["driver"], "Why": rota["why"], "Their notes": rota["notes"]}), key="rota")

    if len(unfilled):
        st.header("Journeys the draft couldn't fill")
        st.markdown("For these, ask the drivers listed: they're trained and live nearby but left that day blank on "
                    "the availability sheet. If nobody can, these are the journeys that would go to a taxi or a "
                    "member of staff.")
        ui.table(pd.DataFrame({
            "Day": unfilled["date"].map(day), "Time": unfilled["time"],
            "Service": unfilled["service"].map(lambda s: SERVICE_NAMES.get(s, s)),
            "From": [end(h, p) for h, p in zip(unfilled["pickup_hospital"], unfilled["pickup_postcode"])],
            "To": [end(h, p) for h, p in zip(unfilled["dropoff_hospital"], unfilled["dropoff_postcode"])],
            "Why it's not filled": unfilled["why"], "Worth asking": unfilled["ask"]}), key="rota_unfilled")

    st.header("Each driver's journeys")
    st.markdown("One list per driver, to copy into their email or print. The download has the full addresses from "
                "your journeys file, for sending to the driver; the page shows postcode areas only.")
    for name, mine in rota.groupby("driver", sort=True):
        with st.expander(f"{name}: {len(mine)} journey{'s' if len(mine) > 1 else ''} on {mine['date'].nunique()} "
                         f"day{'s' if mine['date'].nunique() > 1 else ''}"):
            st.markdown("\n".join(
                f"- **{day(r.date)}{', ' + r.time if r.time else ''}**: {SERVICE_NAMES.get(r.service, r.service)}, "
                f"from {end(r.pickup_hospital, r.pickup_postcode)} to "
                f"{end(r.dropoff_hospital, r.dropoff_postcode)}" for r in mine.itertuples()))
    sheet = pd.DataFrame({
        "Driver": rota["driver"], "Day": rota["date"].map(day), "Time": rota["time"],
        "Service": rota["service"].map(lambda s: SERVICE_NAMES.get(s, s)),
        "From": [a if isinstance(a, str) else end(h, p)
                 for a, h, p in zip(rota["pickup_address"], rota["pickup_hospital"], rota["pickup_postcode"])],
        "To": [a if isinstance(a, str) else end(h, p)
               for a, h, p in zip(rota["dropoff_address"], rota["dropoff_hospital"], rota["dropoff_postcode"])]})
    st.download_button("Download every driver's journeys (opens in Excel)",
                       sheet.sort_values(["Driver", "Day"]).to_csv(index=False).encode(), "driver_journeys.csv")

    with st.expander("How this is worked out"):
        st.markdown(f"""
A driver can be suggested for a journey if they are trained for the service, marked free that day on the
availability sheet (respecting "AM", "No AM journeys" and "one day"), and live within
{net.pickup_reach_km * MILES_PER_KM:.0f} miles of the pickup by road.

The journeys with the fewest suitable drivers are filled first, so the hard ones aren't left until nobody is free.
For each journey, a driver already driving from or to the same hospital that day comes first, then one already
driving that day, then whoever has the fewest days so far that week, then the nearest. Nobody gets more than {cap}
journeys in a day: three in four driver-days in the weeks that record who drove had that many or fewer.

Pickup times are often missing, so the draft can't check that two journeys on the same day don't overlap: check
the times before confirming. It can't read drivers' notes either, so they're shown next to each suggestion.
""")
