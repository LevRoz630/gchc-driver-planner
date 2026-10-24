"""The overview map (both pages) and, on the demo page only, where to recruit.

Same method as notebooks/anas_gaps.ipynb. Uses the Network plus public_data/health_boards.json (board outlines) and
districts.csv (app/build_data.py). The background is Carto's free street map: only the view is sent, never postcodes.
Drivers and families are shown by postcode district, never by their own postcode."""
import json
import numpy as np
import pandas as pd
import pydeck as pdk

import streamlit as st

import ui

from planner import HOSPITAL_NAMES, MILES_PER_KM, SERVICE_NAMES, district, hospitals, km

miles = lambda k: k * MILES_PER_KM   # distances are worked out in km and shown in miles

# every kind of mark has its own shape and colour, so neither is the only cue
KINDS = {"Hospital": ("square", "#1A1A1A"),   # (map shape, colour); the key above the map uses the same
         "Drivers live here": ("circle", "#0B5CAD"),
         "Families' journeys": ("triangle-up", "#B35900"),
         "Families' journeys, no driver nearby": ("diamond", "#B00020")}


# journeys whose non-hospital end is a patient's family; milk bank and breast pump ends are donors and mothers
PATIENT_SERVICES = {"PAT", "RESP", "NNU", "NICU"}


def families(net, service):
    """Patient journeys a week by the postcode district of their non-hospital end (where the family is). Milk bank
    and breast pump journeys are left out: their other end is a donor or a mother, not a family needing a lift."""
    j = net.journeys[net.journeys["service"].isin(PATIENT_SERVICES)]
    j = j if service is None else j[j["service"] == service]
    home = j["pickup_postcode"].where(j["pickup_hospital"].isna(), j["dropoff_postcode"].where(j["dropoff_hospital"].isna()))
    return (district(home).dropna().value_counts() / net.n_weeks).rename("journeys_pw").rename_axis("district")


# corners of each marker shape around (0, 0), radius 1; the map draws them as small filled polygons
SHAPES = {"square": [(-.8, -.8), (.8, -.8), (.8, .8), (-.8, .8)],
          "circle": [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)],
          "triangle-up": [(0, 1.1), (.95, -.6), (-.95, -.6)],
          "diamond": [(0, 1.15), (1.15, 0), (0, -1.15), (-1.15, 0)]}


def outline(shape, lon, lat, metres):
    """A marker shape centred on a point, `metres` across from centre to edge."""
    dlat = metres / 111_320
    dlon = dlat / np.cos(np.radians(lat))
    return [[lon + x * dlon, lat + y * dlat] for x, y in SHAPES[shape]]


SCOTLAND = pdk.ViewState(latitude=57.0, longitude=-4.3, zoom=5.4)


def view(lat, lon):
    """Centre and zoom that fit most of these points (the outer 5% either side can sit off screen); all of Scotland
    when there are none to fit."""
    if not np.isfinite(np.asarray(lat, dtype=float)).any():
        return SCOTLAND
    la, lo = np.nanpercentile(lat, [5, 95]), np.nanpercentile(lon, [5, 95])
    zoom = np.log2(min(360 / max(lo[1] - lo[0], 0.05) * 1.6, 170 / max(la[1] - la[0], 0.03) * 1.2))
    return pdk.ViewState(latitude=float(la.mean()), longitude=float(lo.mean()), zoom=float(min(zoom, 11)) - 0.6)


def render_overview(net, centres, folder):
    hosp = hospitals(net)
    running = [s for s in SERVICE_NAMES if s in set(net.journeys["service"])]
    pick = st.radio("Show journeys and drivers for", ["All services", *running],
                    format_func=lambda s: SERVICE_NAMES.get(s, s), horizontal=True)
    service = None if pick == "All services" else pick
    word = "volunteer" if service is None else f"{SERVICE_NAMES[service].lower()} driver"

    drv = net.drv[net.drv["lat"].notna()].copy()
    if service is not None:
        drv = drv[drv.get(f"role_{service}", pd.Series(False, index=drv.index)).astype(bool)]
        hosp = hosp[hosp["services"].map(lambda s: service in s)]
    drv["district"] = district(drv["postcode"])
    by_area = drv.groupby("district").size().rename("drivers").to_frame().join(centres, how="inner")

    near = float(net.legs["home_km"].median())
    fam = families(net, service).to_frame().join(centres, how="inner")
    d = km(fam["lat"].values[:, None], fam["lon"].values[:, None], drv["lat"].values[None], drv["lon"].values[None],
           net.s.road_factor)
    fam["nearest_km"] = d.min(axis=1) if len(drv) else np.inf
    gap = fam["nearest_km"] > near

    c = st.columns(4)
    jobs = net.journeys if service is None else net.journeys[net.journeys["service"] == service]
    c[0].metric("Journeys a week", f"{len(jobs) / net.n_weeks:.0f}")
    c[1].metric(f"{word.capitalize()}s", len(drv))
    c[2].metric("Hospitals", len(hosp))
    c[3].metric("Family areas far from a driver",
                f"{int(gap.sum())} of {len(fam)}" if len(fam) else "Not a patient service")

    far, close = fam[gap], fam[~gap]   # columns built from each part, so an empty part adds no rows
    marks = pd.concat([
        far.assign(kind="Families' journeys, no driver nearby", n=far["journeys_pw"],
                   detail=far["journeys_pw"].map(lambda v: f"{v:.1f} journeys a week")
                   + far["nearest_km"].map(lambda v: f", nearest {word} {miles(v):.0f} miles")),
        close.assign(kind="Families' journeys", n=close["journeys_pw"],
                     detail=close["journeys_pw"].map(lambda v: f"{v:.1f} journeys a week")),
        by_area.assign(kind="Drivers live here", n=by_area["drivers"],
                       detail=by_area["drivers"].map(lambda v: f"{v} {word}{'s' if v != 1 else ''}")),
    ]).rename_axis("area").reset_index()
    hmarks = hosp.assign(kind="Hospital", area=hosp["name"], n=hosp["journeys_pw"],
                         detail=[f"{r.journeys_pw:.1f} journeys a week: "
                                 + ", ".join(SERVICE_NAMES.get(x, x) for x in r.services) for r in hosp.itertuples()])
    marks = pd.concat([marks, hmarks])[["area", "kind", "n", "detail", "lat", "lon"]]
    # draw order: families underneath, then drivers, hospitals on top
    marks["rank"] = marks["kind"].map({"Families' journeys": 0, "Families' journeys, no driver nearby": 1,
                                       "Drivers live here": 2, "Hospital": 3})

    whole = st.checkbox("Show all of Scotland", value=False)
    start = (SCOTLAND if whole
             else view(marks["lat"].values, marks["lon"].values))
    # marker sizes in pixels at the starting zoom, turned into metres so the shapes can be drawn on the map
    metres_per_px = 156_543 * np.cos(np.radians(start.latitude)) / 2 ** start.zoom
    px = np.where(marks["kind"] == "Hospital", 7, 3 + 5 * np.sqrt(marks["n"] / max(float(marks["n"].max()), 1)))
    marks["shape"] = [outline(KINDS[k][0], lo, la, p * metres_per_px)
                      for k, lo, la, p in zip(marks["kind"], marks["lon"], marks["lat"], px)]
    marks["fill"] = marks["kind"].map(lambda k: [int(KINDS[k][1][i:i + 2], 16) for i in (1, 3, 5)])
    marks = marks.sort_values("rank")     # families underneath, then drivers, hospitals on top
    boards = json.loads((folder / "health_boards.json").read_text())
    layers = [
        pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"board": b["board"]}, "geometry": b["geometry"]} for b in boards]},
            stroked=True, filled=False, get_line_color=[90, 100, 115, 160], line_width_min_pixels=1),
        # one layer per kind, families first, so each kind's white edges sit over the kinds beneath it
        *[pdk.Layer("PolygonLayer", group, get_polygon="shape", get_fill_color="fill", get_line_color=[255, 255, 255],
                    line_width_min_pixels=1.5, stroked=True, filled=True, pickable=True, parameters={"depthTest": False})
          for _, group in marks.groupby("rank")],
        pdk.Layer("TextLayer", hmarks, get_position=["lon", "lat"], get_text="hospital", get_size=13,
                  get_color=[26, 26, 26], get_pixel_offset=[16, 0], get_text_anchor="'start'",
                  font_weight=700, background=True, get_background_color=[255, 255, 255, 230],
                  background_padding=[3, 1]),
    ]
    deck = pdk.Deck(layers=layers, initial_view_state=start, map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
                    tooltip={"html": "<b>{area}</b><br/>{detail}",
                             "style": {"backgroundColor": "#1A1A1A", "color": "white", "fontSize": "14px"}})

    # the key is page text rather than a chart legend, so it's readable at any zoom and by screen readers
    symbol = {"square": "■", "circle": "●", "triangle-up": "▲", "diamond": "◆"}
    st.markdown(" &nbsp; ".join(f'<span style="color:{c}" aria-hidden="true">{symbol[sh]}</span> {k}'
                                for k, (sh, c) in KINDS.items()) + " &nbsp; (bigger marks: more drivers or journeys; zoom with the map's + and − buttons, or scroll or pinch)",
                unsafe_allow_html=True)
    st.pydeck_chart(deck, height=620)

    # the same picture in words, for anyone who can't see the map
    lines = [] if hosp.empty else [f"- **Hospitals:** " + "; ".join(f"{r.name} ({r.hospital}), {r.journeys_pw:.1f} journeys a week"
                                               for r in hosp.itertuples())]
    if len(by_area):
        top = by_area["drivers"].sort_values(ascending=False)
        lines.append(f"- **Where {word}s live:** {len(top)} postcode areas; most in "
                     + ", ".join(f"{a} ({v})" for a, v in top.head(5).items()) + ".")
    if gap.any():
        g = fam[gap].sort_values("journeys_pw", ascending=False)
        lines.append(f"- **Families with no {word} within {miles(near):.0f} miles:** "
                     + ", ".join(f"{a} ({r.journeys_pw:.1f} a week, nearest {miles(r.nearest_km):.0f} miles)"
                                 for a, r in g.head(8).iterrows()) + ".")
    st.markdown("\n".join(lines))
    # every mark on the map with what its hover shows, so nothing is only on the map
    with st.expander("The map as a table"):
        ui.table(marks.sort_values(["rank", "area"], ascending=[False, True]).rename(columns={"kind": "What", "area": "Where", "detail": "Details"})
                 [["What", "Where", "Details"]], key="map_marks")

    st.header("Hospitals and the services we run there")
    reach = net.hospital_reach_km
    trained_near = []
    for r in hosp.itertuples():
        d = km(net.drv["lat"], net.drv["lon"], r.lat, r.lon, net.s.road_factor) <= reach
        roles = [f"role_{s}" for s in r.services if f"role_{s}" in net.drv]
        trained_near.append(int((d & net.drv[roles].any(axis=1)).sum()) if roles else 0)
    ui.table(pd.DataFrame({
        "Hospital": hosp["name"] + " (" + hosp["hospital"] + ")", "Services": hosp["services"].map(lambda s: ", ".join(SERVICE_NAMES.get(x, x) for x in s)),
        "Journeys a week": hosp["journeys_pw"].round(1),
        f"Trained drivers within {miles(reach):.0f} miles": trained_near}), key="hospitals")

    render_activity(net)

    with st.expander("How this is worked out"):
        st.markdown(f"""
Journeys come from every week in the journeys file. Patient transport, respiratory and neonatal journeys are
placed at the postcode area of their family end (the end that isn't a hospital). Milk bank and breast pump journeys
only count at the hospitals, because their other end is a donor or a mother, not a family needing a lift.
Drivers are placed at the postcode area they live in. An area has a driver nearby if one lives within {miles(near):.0f} miles
by road, the distance drivers usually travel to a pickup today. Distances are straight lines times 1.3 for roads.

The map only shows families who already got a lift. Families who needed one but never asked aren't in the data;
'Where should we recruit?' estimates where they are from the census.
""")


def no_driver_nearby(net, districts):
    """Each district's nearest driver, and whether any driver lives within the distance drivers usually travel to
    their first pickup today."""
    near = float(net.legs["home_km"].median())
    drv = net.drv[net.drv["lat"].notna()]
    d = km(districts["lat"].values[:, None], districts["lon"].values[:, None], drv["lat"].values[None],
           drv["lon"].values[None], net.s.road_factor)
    out = districts.copy()
    out["drivers_nearby"] = (d <= near).sum(axis=1)
    out["nearest_km"] = d.min(axis=1)
    pat = drv["role_PAT"].values if "role_PAT" in drv.columns else np.zeros(len(drv), bool)
    out["nearest_pat_km"] = np.where(pat[None], d, np.inf).min(axis=1) if pat.any() else np.nan
    return out, near


def render_recruit(net, districts):
    out, near = no_driver_nearby(net, districts)
    none = out["drivers_nearby"] == 0
    share = (out["need"].where(none, 0).groupby(out["board"]).sum() / out.groupby("board")["need"].sum())
    share = share.sort_values(ascending=False)

    st.markdown(f"Share of each health board's children's need where no volunteer lives within {miles(near):.0f} miles, the "
                "distance drivers usually travel to their first pickup today.")
    c = st.columns(len(share))
    for col, (board, v) in zip(c, share.items()):
        col.metric(board, f"{v:.0%}")

    st.header("Where to recruit")
    top = out[none].sort_values("need", ascending=False).head(15)
    view = pd.DataFrame({
        "Area": top["district"], "Council": top["council"], "Health board": top["board"],
        "Children": top["children"],
        "Nearest driver (miles)": miles(top["nearest_km"]).round(0).astype(int),
        "Nearest patient transport driver (miles)": miles(top["nearest_pat_km"].replace(np.inf, np.nan)).round(0)})
    ui.table(view, key="recruit", column_config={
        "Children": st.column_config.NumberColumn(format="%d"),
        "Nearest patient transport driver (miles)": st.column_config.NumberColumn(format="%d")})
    st.download_button("Download this table (opens in Excel)", view.to_csv(index=False).encode(),
                       "where_to_recruit.csv")

    with st.expander("How this is worked out"):
        st.markdown(f"""
Need in each area is the number of children, times the share whose daily activities are limited by a health
problem or disability, times the share of households without a car (Scotland's Census 2022 and NRS population
estimates). Across Scotland's councils this ranks areas almost exactly as Child Disability Payment does, and car
access tracks missed children's appointments.

An area counts as having a driver nearby if any volunteer in the drivers file lives within {miles(near):.0f} miles of its
centre, by road. It uses where drivers live, not how many journeys an area had, because a few weeks of journeys
are too few to show which areas are missing out.
""")


# ---------- activity: journeys by day, how the work is spread, availability replies ----------
SHADES = ["#FFFFFF", "#DCEBFA", "#B9D6F5", "#8DBDEE", "#5FA2E6"]   # light blue steps; the number is always shown


def shade(v, top):
    if not isinstance(v, (int, float)) or v <= 0 or top <= 0:
        return "background-color: #FFFFFF; color: #1A1A1A"
    return f"background-color: {SHADES[min(4, 1 + int(3 * v / top))]}; color: #1A1A1A"


def activity(net):
    """Journeys by service and weekday, availability replies by week, and journeys per driver."""
    from planner import WEEKDAYS
    j = net.journeys.assign(weekday=pd.to_datetime(net.journeys["date"]).dt.day_name())
    by_day = (j.groupby(["service", "weekday"]).size() / net.n_weeks).unstack(fill_value=0)
    by_day = by_day.reindex(columns=[d for d in WEEKDAYS if d in by_day.columns])

    status = net.availability["status"].astype(str).str.strip().str.lower()
    replies = (pd.crosstab(net.availability["week_start"], status)
               .reindex(columns=["available", "unavailable", "unknown"], fill_value=0))

    driven = net.legs.groupby("driver").agg(journeys=("service", "size"), days=("date", "nunique"))
    free = net.available[net.available["week_start"].isin(net.driver_weeks)].groupby("driver")["date"].nunique()
    spread = pd.DataFrame({"journeys": driven["journeys"], "days_driven": driven["days"], "days_free": free}
                          ).fillna(0).astype(int)
    spread = spread[(spread["journeys"] > 0) | (spread["days_free"] > 0)]
    return by_day, replies, spread.sort_values(["journeys", "days_free"], ascending=False)


def render_activity(net):
    by_day, replies, spread = activity(net)

    st.header("Journeys by service and day")
    st.markdown(f"Average journeys a week, across the {net.n_weeks} weeks in the journeys file.")
    table = by_day.round(1).rename(index=lambda s: SERVICE_NAMES.get(s, s)).rename(columns=lambda d: d[:3])
    table["Week"] = table.sum(axis=1).round(1)
    table = table.sort_values("Week", ascending=False).rename_axis("Service")
    top = float(table.drop(columns="Week").values.max())
    ui.table(table.style.map(lambda v: shade(v, top), subset=table.columns[:-1]).format("{:.1f}"),
             key="by_day", index=True)

    st.header("Is the work spread out?")
    weeks = " and ".join(pd.to_datetime(net.driver_weeks).strftime("%-d %B"))
    unused = spread[(spread["journeys"] == 0) & (spread["days_free"] > 0)]
    top3 = spread["journeys"].head(3).sum() / max(spread["journeys"].sum(), 1)
    st.markdown(f"Week of {weeks}, the latest week that records who drove. {(spread['journeys'] > 0).sum()} "
                f"volunteers drove, and the three busiest did **{top3:.0%}** of the volunteer journeys.")
    if len(unused):
        st.markdown(f"**{len(unused)} were marked free but got no journeys:** {', '.join(unused.index)}. Worth asking "
                    "them why, and offering them the next journeys that suit them.")
    ui.table(pd.DataFrame({"Driver": spread.index, "Journeys": spread["journeys"].values,
                               "Days driven": spread["days_driven"].values,
                               "Days marked free": spread["days_free"].values}),
             key="spread", column_config={
                     "Journeys": st.column_config.ProgressColumn(format="%d", min_value=0,
                                                                 max_value=max(int(spread["journeys"].max()), 1))})

    st.header("Are drivers replying with their availability?")
    view = replies.rename(columns={"available": "Free", "unavailable": "Not free", "unknown": "Left blank"})
    view.index = pd.to_datetime(view.index).strftime("Week of %-d %B")
    view["Left blank (%)"] = (100 * view["Left blank"] / view[["Free", "Not free", "Left blank"]].sum(axis=1)).round(0)
    ui.table(view.rename_axis("Week"), key="replies", index=True, formats={"Left blank (%)": "{:.0f}%"}, column_config={
        "Left blank (%)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    st.markdown("A blank counts as not free, so every blank is a driver the coordinators can't plan on. Asking for "
                "a reply each week (free all day, morning only, afternoon only, or not free), and chasing blanks, "
                "is the quickest way to find more drivers.")
