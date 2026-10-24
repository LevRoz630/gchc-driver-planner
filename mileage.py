"""Mileage and reimbursement, from the optional fourth upload (the yearly mileage report).

The taxi comparison uses the same journeys, fares and mileage rate as the "driver stops volunteering" page
(planner.Dropout)."""
import pandas as pd

import streamlit as st

import ui


def claims(mileage, settings):
    """One row per driver: miles and claim. Claim = claim_gbp if given, else miles x rate_gbp_per_mile if given,
    else miles x the planner's mileage rate."""
    m = mileage.copy()
    m["driver"] = m["driver"].astype(str).str.strip()
    m["miles"] = pd.to_numeric(m["miles"], errors="coerce").fillna(0)
    if "claim_gbp" in m:
        m["claim"] = pd.to_numeric(m["claim_gbp"], errors="coerce")
    else:
        m["claim"] = float("nan")
    rate = pd.to_numeric(m["rate_gbp_per_mile"], errors="coerce") if "rate_gbp_per_mile" in m else settings.mileage_rate
    m["claim"] = m["claim"].fillna(m["miles"] * rate)
    return (m.groupby("driver")[["miles", "claim"]].sum()
              .sort_values("claim", ascending=False))


def render(net, drop, mileage):
    m = claims(mileage, net.s)
    total, miles = m["claim"].sum(), m["miles"].sum()
    claimants = m[m["claim"] > 0]

    c = st.columns(4)
    c[0].metric("Miles claimed", f"{miles:,.0f}")
    c[1].metric("Paid in mileage", f"£{total:,.0f}")
    c[2].metric("Drivers claiming", len(claimants))
    c[3].metric("Average claim per driver", f"£{total / max(len(claimants), 1):,.0f}")

    st.header("What volunteer driving saves")
    week = " and ".join(pd.to_datetime(net.driver_weeks).strftime("%-d %B"))
    ratio = drop.days["taxi_cost"].sum() / max(drop.days["volunteer_cost"].sum(), 1e-9)
    st.markdown(f"In the week of {week}, the volunteer journeys would have cost about **{ratio:.1f} times as much "
                f"by taxi** as the mileage paid for them, even counting the drive to and from each volunteer's home. "
                f"At that rate the £{total:,.0f} paid in mileage stands in for about **£{total * ratio:,.0f} of "
                "taxis** over the year.")

    st.header("Who drives the most miles")
    top5 = claimants["claim"].head(5).sum() / max(total, 1e-9)
    st.markdown(f"The five drivers with the biggest claims account for **{top5:.0%}** of all mileage. Drivers who "
                "cover a lot of miles are the ones it would cost most to lose, and the ones to check in with.")
    jobs = net.legs["driver"].value_counts()
    view = pd.DataFrame({
        "Driver": m.index,
        "Miles": m["miles"].round(0).astype(int).values,
        "Claim (£)": m["claim"].round(0).values,
        "Share of all mileage": (100 * m["claim"] / max(total, 1e-9)).round(1).values,
        f"Journeys in the week of {week}": jobs.reindex(m.index, fill_value=0).astype(int).values,
    })
    ui.table(view, key="mileage", formats={"Claim (£)": "£{:,.0f}", "Share of all mileage": "{:.1f}%"}, column_config={
        "Claim (£)": st.column_config.ProgressColumn(format="£%.0f", min_value=0,
                                                     max_value=float(max(view["Claim (£)"].max(), 1))),
        "Share of all mileage": st.column_config.NumberColumn(format="%.1f%%")})
    st.download_button("Download this table (opens in Excel)", view.to_csv(index=False).encode(), "mileage.csv")

    listed = set(net.drv.index)
    not_listed = sorted(set(claimants.index) - listed)
    drove_no_claim = sorted(set(jobs.index) - set(claimants.index))
    if not_listed or drove_no_claim:
        st.header("Names to check")
        if not_listed:
            st.markdown(f"**Claimed mileage but aren't in the drivers list:** {', '.join(not_listed)}. They may have "
                        "stopped volunteering, or their name is spelled differently.")
        if drove_no_claim:
            st.markdown(f"**Drove in the week of {week} but have no mileage claim:** {', '.join(drove_no_claim)}. "
                        "They may not be claiming what they're owed.")

    with st.expander("How this is worked out"):
        st.markdown(f"""
Each driver's claim comes from the claim column if the file has one, otherwise their miles times the rate column,
otherwise their miles times {net.s.mileage_rate * 100:.0f}p a mile.

The taxi comparison takes the volunteer journeys in the week of {week}, prices each one at Glasgow taxi fares on
the road distance, and compares that with the mileage paid for the same journeys, including the drive from the
volunteer's home to the first pickup and back. The mileage report covers a whole year and the journeys file only
some weeks, so the comparison assumes that week is typical. Long milk bank runs to Dundee, Aberdeen and Inverness
usually go to partner charities rather than taxis, so the true saving is somewhat lower than shown.
""")
