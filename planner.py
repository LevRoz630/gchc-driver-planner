"""The models behind the planner: coverage, new departments and drivers who stop, as plain functions.

Same method as notebooks/lev_expansion_model.ipynb and notebooks/lev_driver_dropout.ipynb; assumptions are
explained in notebooks/lev_expansion_model.ipynb and the dropout notebook's caveats.

Charity inputs (the layout checks.py describes and convert.py writes):
  drivers.csv       driver, postcode, role_<SERVICE> (True/False) for each service
  availability.csv  driver, week_start, date, status (available / unavailable / unknown)
  journeys.csv      week_start, date, service, driver, driver_type (volunteer / relay / taxi),
                    pickup_postcode, dropoff_postcode, pickup_hospital, dropoff_hospital
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MILES_PER_KM = 0.6214
BIRTH_SERVICES = ["MB", "BP", "NNU"]
SCENARIOS = ["low", "central", "high"]
SERVICE_NAMES = {"PAT": "Patient transport", "RESP": "Respiratory", "MB": "Milk bank", "BP": "Breast pumps",
                 "NNU": "Neonatal unit", "MM": "Memory making", "CARDIO": "Cardiology", "BIO": "BIOPIC study"}
SERVICES = set(SERVICE_NAMES)
HOSPITAL_NAMES = {"RHC": "Royal Hospital for Children", "CCF": "Crosshouse", "PRM": "Princess Royal Maternity",
                  "RAH": "Royal Alexandra Hospital", "WISHAW": "University Hospital Wishaw",
                  "FVRH": "Forth Valley Royal Hospital", "RIE": "Royal Infirmary of Edinburgh",
                  "ARI": "Aberdeen Royal Infirmary", "NINEWELLS": "Ninewells Hospital",
                  "RAIGMORE": "Raigmore Hospital", "CDS": "Children's Diabetes Service, Renfrew"}


@dataclass(frozen=True)
class Settings:
    road_factor: float = 1.3        # road km = straight-line km x 1.3
    reach_quantile: float = 0.8     # "within reach" = this quantile of today's home-to-hospital distances
    mileage_rate: float = 0.45      # GBP a mile paid to volunteers
    taxi_flag_fall: float = 4.50    # Glasgow tariff, 2025 review: first 880 yards
    taxi_flag_yards: float = 880
    taxi_step_price: float = 0.30   # then 30p per 174 yards
    taxi_step_yards: float = 174


def km(lat1, lon1, lat2, lon2, road_factor):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a)) * road_factor


def load_public(folder):
    folder = Path(folder)
    return {"candidates": pd.read_csv(folder / "candidates.csv"),
            "districts": pd.read_csv(folder / "catchment_districts.csv"),
            "constants": json.loads((folder / "constants.json").read_text()),
            "coords": pd.read_csv(folder / "postcodes.csv").set_index("pc")}


def place_postcodes(df, col, coords, prefix=""):
    """Add lat/lon (with an optional prefix) from a postcode column, ignoring spaces and case."""
    key = df[col].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    return df.assign(_pc=key).join(coords.add_prefix(prefix), on="_pc").drop(columns="_pc")


def district(postcodes):
    """'G51 4TF' -> 'G51' (the postcode without its last three characters); blank stays blank."""
    key = postcodes.astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    return key.str[:-3].where(postcodes.notna() & (key.str.len() > 4))


def hospitals(net):
    """One row per hospital that journeys start or end at: code, name, services, journeys a week and where it is,
    busiest first."""
    j = net.journeys
    ends = pd.concat([j[["service", "pickup_hospital", "pickup_postcode"]].set_axis(["service", "hospital", "postcode"], axis=1),
                      j[["service", "dropoff_hospital", "dropoff_postcode"]].set_axis(["service", "hospital", "postcode"], axis=1)])
    ends = ends[ends["hospital"].notna()]
    out = ends.groupby("hospital").agg(journeys_pw=("service", "size"), services=("service", lambda s: sorted(set(s))),
                                       postcode=("postcode", lambda p: p.mode().iat[0] if p.notna().any() else None))
    out["journeys_pw"] /= net.n_weeks
    out = net.place(out.reset_index(), "postcode")
    out["name"] = out["hospital"].map(lambda h: HOSPITAL_NAMES.get(h, h))
    return out[out["lat"].notna()].sort_values("journeys_pw", ascending=False)


class Network:
    """The charity's drivers and journeys, placed on the map. Driver figures come from the weeks that record who
    drove (at least half the journeys name a driver)."""

    def __init__(self, drivers, availability, journeys, coords, s: Settings):
        self.s = s
        journeys = journeys.assign(service=journeys["service"].replace({"NICU": "NNU"}))
        journeys = journeys[journeys["service"].isin(SERVICES)]     # the checks warn about any others
        self.journeys = journeys
        self.n_weeks = journeys["week_start"].nunique()
        named = journeys.groupby("week_start")["driver"].apply(lambda d: d.notna().mean())
        self.driver_weeks = sorted(named[named >= 0.5].index)
        if not self.driver_weeks:
            raise ValueError("No week in the journeys file records who drove.")

        self.availability = availability
        place = self.place = lambda df, col, prefix="": place_postcodes(df, col, coords, prefix)

        self.drv = place(drivers.drop_duplicates("driver"), "postcode").set_index("driver")
        self.services = [c[5:] for c in self.drv.columns if c.startswith("role_")]
        for c in self.drv.columns:
            if c.startswith("role_"):
                self.drv[c] = self.drv[c].astype(str).str.lower().isin(["true", "1", "yes"])
        legs = journeys[(journeys["driver_type"] == "volunteer") & journeys["driver"].isin(self.drv.index)
                        & journeys["week_start"].isin(self.driver_weeks)]
        legs = place(place(legs, "pickup_postcode", "from_"), "dropoff_postcode", "to_")
        legs = legs[legs["from_lat"].notna() & legs["to_lat"].notna()].copy()
        f = s.road_factor
        legs["leg_km"] = km(legs["from_lat"], legs["from_lon"], legs["to_lat"], legs["to_lon"], f)
        dl, dn = legs["driver"].map(self.drv["lat"]), legs["driver"].map(self.drv["lon"])
        legs["home_km"] = km(dl, dn, legs["from_lat"], legs["from_lon"], f)
        # the hospital end of each journey, for how far drivers travel to the hospitals they serve
        at_pickup = legs["pickup_hospital"].notna()
        legs["hosp_km"] = km(dl, dn, np.where(at_pickup, legs["from_lat"], legs["to_lat"]),
                             np.where(at_pickup, legs["from_lon"], legs["to_lon"]), f)
        self.legs = legs

        self.available = availability[availability["status"].astype(str).str.strip().str.lower() == "available"]
        week = self.available[self.available["week_start"].isin(self.driver_weeks)]
        n = len(self.driver_weeks)
        self.drv["available_pw"] = week.groupby("driver")["date"].nunique().reindex(self.drv.index, fill_value=0) / n
        used = legs.groupby("driver")["date"].nunique().reindex(self.drv.index, fill_value=0) / n
        self.drv["spare_pw"] = (self.drv["available_pw"] - used).clip(lower=0).astype(float)
        self.journeys_per_day = len(legs) / legs.groupby(["driver", "date"]).ngroups
        free_pw = self.drv.loc[self.drv["available_pw"] > 0, "available_pw"]
        self.days_per_driver = free_pw.mean() if len(free_pw) else 1.0   # no availability that week: assume one day
        self.hospital_reach_km = float(legs["hosp_km"].dropna().quantile(s.reach_quantile))
        self.pickup_reach_km = float(legs["home_km"].dropna().quantile(s.reach_quantile))


# ---------- Today: where cover is thin, and who to train ----------

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIN_JOURNEYS_PW = 2      # service-and-hospital pairs with fewer journeys a week are too small to assess


def coverage(net):
    """For each service at each hospital, and each weekday: journeys a week (all sample weeks) against the
    trained drivers usually free that day who live within reach of the hospital (all availability weeks).

    Returns (cells, rows): one row per service, hospital and weekday; one row per service and hospital."""
    j = net.journeys.copy()
    j["hospital"] = j["pickup_hospital"].fillna(j["dropoff_hospital"])
    j["hospital_postcode"] = j["pickup_postcode"].where(j["pickup_hospital"].notna(), j["dropoff_postcode"])
    j = j[j["hospital"].notna()]
    j["weekday"] = pd.to_datetime(j["date"]).dt.day_name()
    rows = j.groupby(["service", "hospital"]).agg(journeys_pw=("date", "size"),
                                                  postcode=("hospital_postcode", lambda p: p.mode().iat[0] if p.notna().any() else None))
    rows["journeys_pw"] /= net.n_weeks
    rows = net.place(rows[rows["journeys_pw"] >= MIN_JOURNEYS_PW].reset_index(), "postcode")
    rows = rows[rows["lat"].notna()]

    # how often each driver is free on each weekday, across every week of availability. A day they drove counts
    # as free even if the sheet says otherwise: much of the sheet is left blank ("unknown")
    av = pd.concat([net.available[["driver", "date"]], net.legs[["driver", "date"]]]).drop_duplicates()
    av = av.assign(weekday=pd.to_datetime(av["date"]).dt.day_name())
    weeks = net.availability["week_start"].nunique()
    free = av.groupby(["driver", "weekday"])["date"].nunique().unstack(fill_value=0).reindex(
        index=net.drv.index, columns=WEEKDAYS, fill_value=0) / weeks

    # top driver's share of each row's volunteer journeys, in the weeks that record who drove
    legs = net.legs.assign(hospital=net.legs["pickup_hospital"].fillna(net.legs["dropoff_hospital"]))
    share = legs.groupby(["service", "hospital"])["driver"].agg(
        lambda d: (d.value_counts().iat[0] / len(d), d.value_counts().index[0]))

    cells, out_rows = [], []
    demand = j.groupby(["service", "hospital", "weekday"]).size() / net.n_weeks
    for r in rows.itertuples():
        near = km(net.drv["lat"], net.drv["lon"], r.lat, r.lon, net.s.road_factor) <= net.hospital_reach_km
        trained = net.drv.get(f"role_{r.service}", pd.Series(False, index=net.drv.index)).astype(bool)
        pool = net.drv.index[near & trained]
        weak_days = []
        for day in WEEKDAYS:
            jpw = demand.get((r.service, r.hospital, day), 0.0)
            supply = free.loc[pool, day].sum()     # trained drivers free on a typical week
            if jpw == 0:
                status = ""
            elif supply == 0:
                status = "No one free"
            elif supply < 1:
                status = "Not every week"
            elif supply < 2:
                status = "Relies on one driver"
            else:
                status = "Covered"
            if status not in ("", "Covered"):
                weak_days.append(day)
            cells.append({"service": r.service, "hospital": r.hospital, "weekday": day, "journeys_pw": jpw,
                          "drivers_free": supply, "status": status})
        top = share.get((r.service, r.hospital))
        run_days = [day for day in WEEKDAYS if demand.get((r.service, r.hospital, day), 0) > 0]
        concentrated = bool(top) and top[0] >= 0.5
        out_rows.append({"service": r.service, "hospital": r.hospital, "journeys_pw": r.journeys_pw,
                         "trained_nearby": len(pool), "weak_days": weak_days,
                         # days to find trainees for: the weak ones, or every day it runs if one driver does
                         # most of the work
                         "train_days": weak_days or (run_days if concentrated else []),
                         "top_driver": top[1] if top else "", "top_share": top[0] if top else np.nan,
                         "lat": r.lat, "lon": r.lon})
    return pd.DataFrame(cells), pd.DataFrame(out_rows), free


def trainees(net, row, free, n=5):
    """Drivers to train for one service at one hospital: not trained for it yet, living within reach, and free
    on the days that need cover. Ranked by how often they're free on those days, then by distance."""
    days = row["train_days"]
    if not days:
        return pd.DataFrame()
    d = km(net.drv["lat"], net.drv["lon"], row["lat"], row["lon"], net.s.road_factor)
    trained = net.drv.get(f"role_{row['service']}", pd.Series(False, index=net.drv.index)).astype(bool)
    ok = (d <= net.hospital_reach_km) & ~trained
    out = pd.DataFrame({
        "driver": net.drv.index[ok],
        "district": net.drv.loc[ok, "postcode_district"] if "postcode_district" in net.drv else "",
        "km": d[ok].round(0),
        "days_free": free.loc[ok, days].sum(axis=1).values,
        "days": [", ".join(day for day in days if free.at[name, day] > 0)
                 for name in net.drv.index[ok]],
        "trained_for": [", ".join(sv for sv in net.services if net.drv.at[name, f"role_{sv}"])
                        for name in net.drv.index[ok]],
        "notes": net.drv.loc[ok, "notes"].fillna("") if "notes" in net.drv else "",
    })
    out = out[out["days_free"] > 0]
    return out.sort_values(["days_free", "km"], ascending=[False, True]).head(n)


# ---------- A new department or maternity unit ----------

def take_up_rates(net, constants):
    per_year = net.journeys["service"].value_counts() * 52 / net.n_weeks
    return {"PAT": {k: per_year.get("PAT", 0) / v for k, v in constants["pat_base"].items()},
            "RESP": per_year.get("RESP", 0) / constants["resp_base"],
            **{b: per_year.get(b, 0) / constants["births_today"] for b in BIRTH_SERVICES}}


def expansion(net, public, candidate_id, scenario="central"):
    """Journeys a week for one candidate and how they'd be covered: spare days, drivers to train, recruits.
    `scenario` picks which patient transport estimate the drivers are sized on (the others are shown as a range)."""
    c = public["candidates"].set_index("candidate").loc[candidate_id]
    k = public["constants"]
    rates = take_up_rates(net, k)
    if c["kind"] == "maternity unit":
        per_year = {b: {sc: c["base"] * rates[b] for sc in SCENARIOS} for b in BIRTH_SERVICES}
    else:
        adjust = c["no_car"] / k["today_no_car"]
        per_year = {"PAT": {sc: c["base"] * rates["PAT"][sc] * adjust for sc in SCENARIOS}}
        if c["location"] != k["rhc_location"]:
            per_year["RESP"] = {sc: c["base"] * k["resp_share"] * rates["RESP"] * adjust for sc in SCENARIOS}

    pool = net.drv.copy()
    in_reach = km(pool["lat"], pool["lon"], c["lat"], c["lon"], net.s.road_factor) <= net.hospital_reach_km
    rows, used_all, train_all = [], set(), set()
    for service, est in per_year.items():
        weekly = {sc: v / 52 for sc, v in est.items()}
        left = weekly[scenario] / net.journeys_per_day
        trained = pool.get(f"role_{service}", pd.Series(False, index=pool.index)).astype(bool)
        used, to_train = [], []
        for group, names in ((used, pool.index[in_reach & trained]), (to_train, pool.index[in_reach & ~trained])):
            for name in pool.loc[names, "spare_pw"].sort_values(ascending=False).index:
                if left <= 1e-9 or pool.at[name, "spare_pw"] <= 0:
                    break
                take = min(left, pool.at[name, "spare_pw"])
                pool.at[name, "spare_pw"] -= take
                pool.at[name, f"role_{service}"] = True
                left -= take
                group.append(name)
        used_all.update(used)
        train_all.update(to_train)
        rows.append({"service": service, **{f"journeys_pw_{sc}": v for sc, v in weekly.items()},
                     "driver_days_pw": weekly[scenario] / net.journeys_per_day, "from_spare_days": ", ".join(used),
                     "to_train": ", ".join(to_train), "recruit_driver_days_pw": left})
    by_service = pd.DataFrame(rows)
    recruit_days = by_service["recruit_driver_days_pw"].sum()
    d = public["districts"]
    d = d[(d["candidate"] == candidate_id) & (d["km"] * net.s.road_factor <= net.hospital_reach_km)]
    return {
        "candidate": c["label"], "health_board": c["health_board"], "kind": c["kind"],
        "already_served": bool(c["served_today"]) and c["kind"] == "maternity unit",
        "services": "+".join(per_year),
        **{f"journeys_pw_{sc}": by_service[f"journeys_pw_{sc}"].sum() for sc in SCENARIOS},
        "drivers_needed": by_service["driver_days_pw"].sum() / net.days_per_driver,
        "from_spare_days": sorted(used_all - train_all), "to_train": sorted(train_all),
        "to_recruit": int(np.ceil(recruit_days / net.days_per_driver - 1e-9)),
        "recruit_districts": list(d.nlargest(3, "weight")["district"]) if recruit_days > 1e-9 else [],
        "by_service": by_service,
    }


def expansion_table(net, public, scenario="central"):
    rows = []
    for cid in public["candidates"]["candidate"]:
        r = expansion(net, public, cid, scenario)
        rows.append({"candidate": r["candidate"], "health_board": r["health_board"], "services": r["services"],
                     "journeys_pw_low": r["journeys_pw_low"], "journeys_pw_central": r["journeys_pw_central"],
                     "journeys_pw_high": r["journeys_pw_high"], "drivers_needed": r["drivers_needed"],
                     "from_spare_days": len(r["from_spare_days"]), "to_train": len(r["to_train"]),
                     "to_recruit": r["to_recruit"], "recruit_districts": ", ".join(r["recruit_districts"]),
                     "already_served": r["already_served"]})
    return pd.DataFrame(rows).sort_values(f"journeys_pw_{scenario}", ascending=False).round(2)


# ---------- What it costs when drivers stop ----------

class Dropout:
    """Replays the driver weeks without the leavers. A stand-in must be available that day, not already driving,
    trained for every service the leaver did that day and within reach of the first pickup; the nearest one takes
    the whole day. Otherwise the day goes by taxi."""

    def __init__(self, net: Network):
        s, legs, drv = net.s, net.legs, net.drv
        yards = legs["leg_km"] * 1093.6
        legs = legs.assign(taxi_cost=s.taxi_flag_fall + np.ceil(
            np.clip(yards - s.taxi_flag_yards, 0, None) / s.taxi_step_yards) * s.taxi_step_price)
        days = legs.groupby(["driver", "date"]).agg(
            services=("service", set), lat=("from_lat", "first"), lon=("from_lon", "first"),
            journeys=("service", "size"), leg_km=("leg_km", "sum"), home_km=("home_km", "first"),
            taxi_cost=("taxi_cost", "sum"))
        days["volunteer_cost"] = (days["leg_km"] + 2 * days["home_km"]) * MILES_PER_KM * s.mileage_rate
        self.days, self.legs = days, legs
        self.cost_per_km = MILES_PER_KM * s.mileage_rate
        self.per_year = 52 / len(net.driver_weeks)
        placed = drv[drv["lat"].notna()]
        names = placed.index.to_numpy()
        free = net.available.groupby("date")["driver"].agg(set)
        busy = pd.Series(days.index.get_level_values("driver"), index=days.index.get_level_values("date")).groupby(level=0).agg(set)
        roles = {c[5:]: placed[c].to_numpy(bool) for c in placed.columns if c.startswith("role_")}

        # for each driver-day: possible stand-ins, nearest first (doesn't depend on who leaves); array maths keeps
        # this quick with hundreds of drivers
        self.options = {}
        for (leaver, date), day in zip(days.index, days.itertuples()):
            dist = km(placed["lat"].to_numpy(), placed["lon"].to_numpy(), day.lat, day.lon, s.road_factor)
            ok = (dist <= net.pickup_reach_km) & (names != leaver)
            ok &= np.isin(names, list(free.get(date, ()))) & ~np.isin(names, list(busy.get(date, ())))
            for x in day.services:
                if x in roles:
                    ok &= roles[x]
            idx = np.flatnonzero(ok)
            self.options[(leaver, date)] = sorted(zip(dist[idx].tolist(), names[idx].tolist()))
        self.active = sorted(legs["driver"].unique())
        # each driver's days, in the order the replay walks them
        cols = ["journeys", "taxi_cost", "volunteer_cost", "home_km", "services"]
        self.by_driver = {n: list(zip(g.index.get_level_values("date"), *(g[c].tolist() for c in cols)))
                          for n, g in days[cols].groupby(level="driver")}
        self._ranking, self._turnover = None, {}

    def replay(self, leavers, detail=False):
        taken, covered, to_taxi, cost, priced, log = set(), 0, 0, 0.0, 0, []
        for leaver in sorted(leavers):
            for date, journeys, taxi_cost, volunteer_cost, home_km, services in self.by_driver.get(leaver, ()):
                pick = next(((d, n) for d, n in self.options[(leaver, date)]
                             if n not in leavers and (n, date) not in taken), None)
                if pick is None:
                    to_taxi += journeys
                    extra = taxi_cost - volunteer_cost
                else:
                    taken.add((pick[1], date))
                    covered += journeys
                    extra = 2 * (pick[0] - home_km) * self.cost_per_km
                if extra == extra:              # a missing postcode leaves that day's cost unknown
                    cost, priced = cost + extra, priced + 1
                if detail:
                    log.append({"driver_leaving": leaver, "date": date, "services": "+".join(sorted(services)),
                                "journeys": journeys, "covered_by": pick[1] if pick else "taxi", "extra_cost": extra})
        out = {"journeys_covered_py": covered * self.per_year, "journeys_to_taxi_py": to_taxi * self.per_year,
               "extra_cost_py": cost * self.per_year if priced or not to_taxi + covered else float("nan")}
        log = pd.DataFrame(log, columns=["driver_leaving", "date", "services", "journeys", "covered_by", "extra_cost"])
        return (out, log) if detail else out

    def ranking(self):
        if self._ranking is not None:
            return self._ranking
        r = pd.DataFrame({n: self.replay({n}) for n in self.active}).T
        r["journeys_py"] = self.legs["driver"].value_counts() * self.per_year
        r["services"] = self.legs.groupby("driver")["service"].agg(lambda x: "+".join(sorted(set(x))))
        r.index.name = "driver"
        self._ranking = r.sort_values("extra_cost_py", ascending=False)
        return self._ranking

    def turnover(self, rate, draws=500, seed=29):
        if (rate, draws) in self._turnover:
            return self._turnover[(rate, draws)]
        rng = np.random.default_rng(seed)
        k = max(1, round(rate * len(self.active)))
        runs = pd.DataFrame([self.replay(set(rng.choice(self.active, k, replace=False))) for _ in range(draws)])
        self._turnover[(rate, draws)] = k, runs
        return k, runs
