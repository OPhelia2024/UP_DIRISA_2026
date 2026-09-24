"""Shared ward-level aggregation logic."""

import pandas as pd


def to_ward_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse voting-station x party rows to one row per ward.

    Filters to the 'Ward' ballot type before aggregating, since PR and
    DC 40% ballots are separate ballots a voter casts and would otherwise
    inflate — and inconsistently inflate, since DC 40% only exists in
    local municipalities, not metros — the vote totals if summed in
    alongside Ward votes.
    """
    df = df[df["BallotType"] == "Ward"].copy()

    # Collapse to one row per voting station first, since RegisteredVoters
    # and SpoiltVotes are duplicated across every party row for a station.
    station = df.groupby(
        ["Province", "Municipality", "Ward", "VotingDistrict"], as_index=False
    ).agg(
        RegisteredVoters=("RegisteredVoters", "first"),
        SpoiltVotes=("SpoiltVotes", "first"),
        TotalValidVotes=("TotalValidVotes", "sum"),  # sum across ward candidates
    )

    # Then aggregate stations up to ward level.
    ward = station.groupby(
        ["Province", "Municipality", "Ward"], as_index=False
    ).agg(
        RegisteredVoters=("RegisteredVoters", "sum"),
        SpoiltVotes=("SpoiltVotes", "sum"),
        TotalValidVotes=("TotalValidVotes", "sum"),
    )
    ward["VotesCast"] = ward["TotalValidVotes"] + ward["SpoiltVotes"]
    ward["Turnout"] = ward["VotesCast"] / ward["RegisteredVoters"]
    return ward


def assert_ward_matches_raw(raw_df: pd.DataFrame, ward_df: pd.DataFrame, ward_id: str) -> None:
    """Regression test: recompute one ward directly from raw rows and
    compare it against the output of to_ward_level for the same ward.

    Run this immediately after every call to to_ward_level, for every
    year. It exists specifically because a missing ballot-type filter
    can silently pass every other consistency check (RegisteredVoters
    and SpoiltVotes consistency, duplicate-row checks, schema checks)
    while still producing wrong vote totals — this is the one check
    that catches it directly, in seconds, rather than requiring a full
    manual investigation.
    """
    raw = raw_df[(raw_df["Ward"] == ward_id) & (raw_df["BallotType"] == "Ward")]
    expected_votes = (
        raw["TotalValidVotes"].sum()
        + raw.groupby("VotingDistrict")["SpoiltVotes"].first().sum()
    )
    actual_votes = ward_df.loc[ward_df["Ward"] == ward_id, "VotesCast"].iloc[0]
    assert abs(expected_votes - actual_votes) < 1, (
        f"Mismatch for {ward_id}: expected {expected_votes}, got {actual_votes}. "
        f"Check that to_ward_level's BallotType filter is present and correct."
    )
    print(f"Sanity check passed for {ward_id}.")