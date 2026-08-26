from collections import Counter
import math
import unicodedata

import numpy as np
import pandas as pd

import config


CITY_GROUP = ["NOMBRE", "Subcanal2"]
STRATEGY_GROUP = ["NOMBRE", "Subcanal2", "ESTRATEGICA"]


def _is_titan(value) -> bool:
    text = unicodedata.normalize("NFKD", str(value).strip().upper())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text in {"TITAN", "TITAN PLUS"}


def _is_specialized(value) -> bool:
    return str(value).strip().upper() in {"SI", "SÍ", "1", "1.0", "TRUE", "ESPECIALIZADA"}


def _segment(value) -> str:
    value = str(value).strip().upper()
    return next((code for code in ("EG", "GR", "ME", "PE") if value.startswith(code)), value)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result[config.SELECTION_COL] = "NO"
    result["_segment"] = result["ESTRATEGICA"].map(_segment)
    result["_fenix"] = pd.to_numeric(result["Fenix"], errors="coerce").fillna(0)
    result["_is_titan"] = result["Programa de Valor"].map(_is_titan)
    result["_mandatory"] = result["_is_titan"] | result["_fenix"].gt(0)
    result["_prev_t"] = result["SEL_MES_ANT"].astype(str).str.strip().str.upper().eq("T")
    result["_is_specialized"] = result["ESPECIALIZADA"].map(_is_specialized)
    result["_selection_channel"] = np.where(
        result["Subcanal2"].eq("ON") | result["CANAL"].eq("ON"), "ON", "OFF"
    )
    # RUT.COM is the physical route key. NOMBRE cannot be part of this key
    # because the source deliberately contains a NOMBRE=0 quota bucket for
    # rows belonging to the same route.
    result["_off_qualified_route"] = result.groupby(
        ["_selection_channel", "RUT.COM"], dropna=False
    )["_mandatory"].transform("any") & result["_selection_channel"].eq("OFF")
    return result


def _historical_targets(historical: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    titles = historical[historical["ES_TITULAR_HISTORICO"]].copy()
    titles["ESTRATEGICA"] = titles["ESTRATEGICA"].map(_segment)
    return (
        titles.groupby(CITY_GROUP).size().astype(int),
        titles.groupby(STRATEGY_GROUP).size().astype(int),
    )


def _with_fixed_on_quotas(historical_targets: pd.Series) -> pd.Series:
    """Replace historical ON totals with the exact approved city quotas."""
    off_values = {
        (str(city), str(subchannel)): int(value)
        for (city, subchannel), value in historical_targets.items()
        if str(subchannel) != "ON"
    }
    values = {
        **off_values,
        **{(city, "ON"): int(quota) for city, quota in config.ON_CITY_QUOTAS.items()},
    }
    keys = sorted(values)
    index = pd.MultiIndex.from_tuples(keys, names=CITY_GROUP)
    return pd.Series([values[key] for key in keys], index=index, dtype=int)


def _city_group_bounds(
    df_elegible: pd.DataFrame,
    historical_targets: pd.Series,
    lima_quotas: dict[str, int],
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Return current capacity and mandatory minimums for every historical/current city."""
    prepared = _prepare(df_elegible)
    capacity_mask = prepared["_selection_channel"].eq("ON") | prepared["_off_qualified_route"]
    capacity_series = prepared.loc[capacity_mask].groupby(CITY_GROUP).size().astype(int)
    mandatory_series = prepared.loc[prepared["_mandatory"]].groupby(CITY_GROUP).size().astype(int)

    lower_bounds = {key: int(mandatory_series.get(key, 0)) for key in capacity_series.index}
    lima_key = ("LIMA", "OFF")
    lima = prepared.loc[
        prepared["NOMBRE"].eq("LIMA")
        & prepared["Subcanal2"].eq("OFF")
        & prepared["_off_qualified_route"]
    ]
    if not lima.empty:
        quota_locations = set(lima_quotas)
        required_lima = int((lima["_mandatory"] & ~lima["LOC.COM"].isin(quota_locations)).sum())
        for location, minimum in lima_quotas.items():
            location_rows = lima.loc[lima["LOC.COM"].eq(location)]
            available = len(location_rows)
            mandatory = int(location_rows["_mandatory"].sum())
            required_lima += max(mandatory, min(int(minimum), available))
        lower_bounds[lima_key] = max(lower_bounds.get(lima_key, 0), required_lima)

    groups = sorted(set(capacity_series.index.to_list()) | set(historical_targets.index.to_list()))
    capacity = {key: int(capacity_series.get(key, 0)) for key in groups}
    lower = {
        key: min(capacity[key], int(lower_bounds.get(key, 0)))
        for key in groups
    }
    return groups, capacity, lower


def _anchored_historical_targets(
    df_elegible: pd.DataFrame,
    historical_targets: pd.Series,
    lima_quotas: dict[str, int],
) -> pd.Series:
    """Keep each prior city quota, capped only by its own current universe.

    A city absent from the current universe therefore becomes zero and its
    missing quota is never transferred to another city. Mandatory Titán/Fénix
    points and Lima minimums remain the only allowed lower-bound exception.
    """
    groups, capacity, lower = _city_group_bounds(
        df_elegible, historical_targets, lima_quotas
    )
    for city, quota in config.ON_CITY_QUOTAS.items():
        key = (city, "ON")
        available = int(capacity.get(key, 0))
        mandatory = int(lower.get(key, 0))
        if available < int(quota):
            raise ValueError(
                f"La cuota ON fija de {city} requiere {int(quota):,} puntos, "
                f"pero la preselección solo tiene {available:,} elegibles ON."
            )
        if mandatory > int(quota):
            raise ValueError(
                f"{city} tiene {mandatory:,} puntos ON obligatorios Titán/Fénix, "
                f"por encima de su cuota fija de {int(quota):,}."
            )
    values = {
        key: max(lower[key], min(capacity[key], int(historical_targets.get(key, 0))))
        for key in groups
    }
    index = pd.MultiIndex.from_tuples(groups, names=CITY_GROUP)
    return pd.Series([values[key] for key in groups], index=index, dtype=int)


def _scaled_city_targets(
    df_elegible: pd.DataFrame,
    historical_targets: pd.Series,
    total_target: int,
    lima_quotas: dict[str, int],
    increase_scope: str = "AMBOS",
) -> pd.Series:
    """Anchor prior city quotas, then allocate only the requested increase.

    The selected scope receives the full change versus the previous month; the
    other channel stays at each city's effective historical quota. A city with
    no current universe is zero and its missing quota is not redistributed.
    Titán/Fénix points and Lima locality minimums act as lower bounds, while the
    current eligible universe is the upper bound.
    """
    total_target = int(total_target)
    if total_target <= 0:
        raise ValueError("La muestra total debe ser un número entero mayor que cero.")
    increase_scope = str(increase_scope).strip().upper()
    if increase_scope not in {"OFF", "ON", "AMBOS"}:
        raise ValueError("El canal del aumento debe ser OFF, ON o AMBOS.")

    groups, capacity, lower = _city_group_bounds(
        df_elegible, historical_targets, lima_quotas
    )
    minimum_total = sum(lower.values())
    maximum_total = sum(capacity.values())
    if total_target < minimum_total:
        raise ValueError(
            f"La muestra solicitada ({total_target:,}) es menor que los {minimum_total:,} "
            "titulares obligatorios requeridos por Titán/Fénix y las cuotas mínimas de Lima."
        )
    if total_target > maximum_total:
        raise ValueError(
            f"La muestra solicitada ({total_target:,}) supera los {maximum_total:,} puntos elegibles disponibles."
        )

    baseline = _anchored_historical_targets(
        df_elegible, historical_targets, lima_quotas
    ).to_dict()
    for city, quota in config.ON_CITY_QUOTAS.items():
        baseline[(city, "ON")] = int(quota)
    baseline_total = int(sum(baseline.values()))
    if total_target < baseline_total:
        raise ValueError(
            f"La muestra solicitada ({total_target:,}) es menor que la base efectiva "
            f"del mes anterior ({baseline_total:,})."
        )

    if total_target > baseline_total and increase_scope != "OFF":
        raise ValueError(
            f"Las cuotas ON son fijas y deben sumar {int(config.ON_FIXED_TOTAL):,}. "
            "Aplica cualquier aumento de la muestra únicamente al canal OFF."
        )

    if increase_scope == "OFF":
        variable_groups = [key for key in groups if key[1] != "ON"]
    elif increase_scope == "ON":
        variable_groups = [key for key in groups if key[1] == "ON"]
    else:
        variable_groups = list(groups)

    targets = {key: int(baseline[key]) for key in groups}
    remaining = total_target - baseline_total
    available_in_scope = sum(capacity[key] - targets[key] for key in variable_groups)
    if remaining > available_in_scope:
        raise ValueError(
            f"El canal seleccionado solo dispone de {available_in_scope:,} puntos adicionales, "
            f"pero se requieren {remaining:,}."
        )

    # Distribute only the increment. Historical quotas already placed above
    # are never recalculated, so an absent city cannot alter another city's base.
    while remaining > 0:
        candidates = [key for key in variable_groups if targets[key] < capacity[key]]
        if not candidates:
            break
        weights = {
            key: max(int(historical_targets.get(key, 0)), int(baseline[key]), 1)
            for key in candidates
        }
        weight_total = sum(weights.values())
        raw = {key: remaining * weights[key] / weight_total for key in candidates}
        additions = {
            key: min(capacity[key] - targets[key], int(math.floor(raw[key])))
            for key in candidates
        }
        granted = sum(additions.values())
        if granted:
            for key, addition in additions.items():
                targets[key] += addition
            remaining -= granted
            continue
        key = max(
            candidates,
            key=lambda value: (
                raw[value], weights[value], capacity[value] - targets[value], value
            ),
        )
        targets[key] += 1
        remaining -= 1

    index = pd.MultiIndex.from_tuples(groups, names=CITY_GROUP)
    return pd.Series([targets[key] for key in groups], index=index, dtype=int)


def _rank(df: pd.DataFrame, prefer_specialized: bool = False) -> pd.DataFrame:
    columns = ["_is_titan", "_mandatory"]
    ascending = [False, False]
    if prefer_specialized:
        columns.append("_is_specialized")
        ascending.append(False)
    if "_preferred_on_route" in df.columns:
        columns.append("_preferred_on_route")
        ascending.append(False)
    columns.extend(["Esparta_Flag", "_prev_t", "Ventas", "CODIGO"])
    ascending.extend([False, True, False, True])
    return df.sort_values(by=columns, ascending=ascending, kind="stable")


def _set_selected(df: pd.DataFrame, index) -> None:
    index = list(index)
    if index:
        df.loc[index, config.SELECTION_COL] = "T"


def _take(df: pd.DataFrame, mask: pd.Series, count: int, prefer_specialized: bool = False) -> int:
    if count <= 0:
        return 0
    available = df.loc[mask & df[config.SELECTION_COL].eq("NO")]
    if available.empty:
        return 0
    chosen = _rank(available, prefer_specialized).index[:count]
    _set_selected(df, chosen)
    return len(chosen)


def _strategy_targets_for(city: str, subchannel: str, strategy_targets: pd.Series) -> dict[str, int]:
    result = {segment: 0 for segment in ("EG", "GR", "ME", "PE")}
    for (target_city, target_subchannel, segment), target in strategy_targets.items():
        if target_city == city and target_subchannel == subchannel:
            result[segment] = int(target)
    return result


def _strategy_percentage_bands(historical: dict[str, int], total_target: int) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return the historical mix target and the closest feasible +/- 1 pp bounds.

    The target is calculated proportionally because Lima or mandatory PDVs can
    make the current total different from the historical total.  The integer
    bounds are adjusted only when a literal one-point interval has no integer
    solution for a very small group.
    """
    segments = ("EG", "GR", "ME", "PE")
    historical_total = sum(historical.values())
    if total_target <= 0 or historical_total <= 0:
        zeros = {segment: 0 for segment in segments}
        return zeros, zeros.copy(), zeros.copy()

    shares = {segment: historical.get(segment, 0) / historical_total for segment in segments}
    raw = {segment: shares[segment] * total_target for segment in segments}
    desired = {segment: int(math.floor(raw[segment])) for segment in segments}
    for segment in sorted(segments, key=lambda value: (raw[value] - desired[value], shares[value]), reverse=True)[:total_target - sum(desired.values())]:
        desired[segment] += 1

    lower = {segment: int(math.ceil(max(0, shares[segment] - 0.01) * total_target)) for segment in segments}
    upper = {segment: int(math.floor(min(1, shares[segment] + 0.01) * total_target)) for segment in segments}

    # Rounding can make small groups have no exact integer solution.  In that
    # case, relax by the minimum one PDV necessary, centred on the historical mix.
    while sum(lower.values()) > total_target:
        candidates = [segment for segment in segments if lower[segment] > 0]
        if not candidates:
            break
        segment = min(candidates, key=lambda value: (desired[value] - lower[value], shares[value]))
        lower[segment] -= 1
    while sum(upper.values()) < total_target:
        segment = max(segments, key=lambda value: (desired[value] - upper[value], shares[value]))
        upper[segment] += 1
    return desired, lower, upper


def _fill_city_total(
    df: pd.DataFrame,
    city: str,
    subchannel: str,
    total_target: int,
    strategy_targets: pd.Series,
    primary_mask: pd.Series,
    fallback_mask: pd.Series | None,
    prefer_specialized: bool,
) -> None:
    """Meet the total while keeping each strategy within +/- 1 pp when feasible."""
    group = df["NOMBRE"].eq(city) & df["Subcanal2"].eq(subchannel)
    historical_reference = _strategy_targets_for(city, subchannel, strategy_targets)
    desired, _lower, upper = _strategy_percentage_bands(historical_reference, total_target)

    # First, complete the proportional historical mix with the preferred
    # candidate universe (Titan routes for OFF; all eligible PDVs for ON).
    for segment, target in desired.items():
        current = int((group & df["_segment"].eq(segment) & df[config.SELECTION_COL].eq("T")).sum())
        remaining_total = total_target - int((group & df[config.SELECTION_COL].eq("T")).sum())
        _take(
            df,
            group & df["_segment"].eq(segment) & primary_mask,
            min(target - current, max(0, remaining_total)),
            prefer_specialized,
        )

    def fill_remaining(candidate_mask: pd.Series, within_band: bool) -> None:
        needed = total_target - int((group & df[config.SELECTION_COL].eq("T")).sum())
        if needed <= 0:
            return
        selected_by_segment = {
            segment: int((group & df["_segment"].eq(segment) & df[config.SELECTION_COL].eq("T")).sum())
            for segment in desired
        }
        # Fill the categories below their historical target first, then use
        # only their remaining one-point capacity.  This avoids a short EG/GR
        # category being silently replaced by a large ME/PE increase.
        for limit in (desired, upper if within_band else None):
            for segment in sorted(desired, key=lambda value: desired[value] - selected_by_segment[value], reverse=True):
                needed = total_target - int((group & df[config.SELECTION_COL].eq("T")).sum())
                if needed <= 0:
                    return
                ceiling = limit[segment] if limit is not None else needed + selected_by_segment[segment]
                capacity = max(0, ceiling - selected_by_segment[segment])
                if capacity <= 0:
                    continue
                taken = _take(
                    df,
                    group & df["_segment"].eq(segment) & candidate_mask,
                    min(capacity, needed),
                    prefer_specialized,
                )
                selected_by_segment[segment] += taken

    fill_remaining(primary_mask, within_band=True)
    # Use a secondary universe only when the caller explicitly provides one.
    # OFF passes no fallback, so it never leaves qualified Titán/Fénix routes.
    if fallback_mask is not None:
        fill_remaining(fallback_mask, within_band=True)

    # A historical total still has priority.  Leave a strategy band only when
    # the eligible universe cannot complete the group inside its +/- 1 pp cap;
    # the Controls sheet exposes every such exception for review.
    fill_remaining(primary_mask, within_band=False)
    if fallback_mask is not None:
        fill_remaining(fallback_mask, within_band=False)


def _ensure_lima_minimums(df_off: pd.DataFrame, lima_quotas: dict[str, int]) -> None:
    """Each Lima locality is a minimum; the city total can add PDVs afterwards."""
    for location, minimum in lima_quotas.items():
        location_mask = (
            df_off["NOMBRE"].eq("LIMA")
            & df_off["LOC.COM"].eq(location)
            & df_off["_off_qualified_route"]
        )
        current = int((location_mask & df_off[config.SELECTION_COL].eq("T")).sum())
        needed = int(minimum) - current
        if needed > 0:
            _take(df_off, location_mask, needed)
        current = int((location_mask & df_off[config.SELECTION_COL].eq("T")).sum())
        if current < int(minimum):
            raise ValueError(
                f"La cuota mínima de Lima para {location} requiere {int(minimum):,} puntos, "
                f"pero solo hay {current:,} disponibles en rutas OFF con Titán o Fénix."
            )


def select_canal_off(
    df_off: pd.DataFrame,
    city_targets: pd.Series,
    strategy_targets: pd.Series,
    lima_quotas: dict[str, int],
    preferred_on_routes: set[str] | None = None,
) -> pd.DataFrame:
    df_off = _prepare(df_off)
    preferred_on_routes = preferred_on_routes or set()
    df_off["_preferred_on_route"] = df_off["RUT.COM"].astype(str).isin(preferred_on_routes)

    # Rule 1: every Titán, Titán Plus or Fénix > 0 is a mandatory titular.
    # This has priority even if it exceeds a historical city/subchannel quota.
    _set_selected(df_off, df_off.index[df_off["_mandatory"]])
    _ensure_lima_minimums(df_off, lima_quotas)

    off_targets = city_targets[city_targets.index.get_level_values("Subcanal2") != "ON"]
    for (city, subchannel), target in off_targets.items():
        _fill_city_total(
            df_off, city, subchannel, int(target), strategy_targets,
            df_off["_off_qualified_route"], None, False,
        )
        actual = int((
            df_off["NOMBRE"].eq(city)
            & df_off["Subcanal2"].eq(subchannel)
            & df_off[config.SELECTION_COL].eq("T")
        ).sum())
        if actual != int(target):
            raise ValueError(
                f"La cuota OFF de {city} ({int(target):,}) no cabe completamente en rutas "
                f"que tengan al menos un punto Titán o Fénix; solo se pudieron asignar {actual:,}."
            )

    _rebalance_off_route_overlap(df_off, preferred_on_routes, lima_quotas)
    _assign_substitutes(df_off)
    return df_off.drop(columns=[column for column in df_off.columns if column.startswith("_")], errors="ignore")


def _rebalance_off_route_overlap(
    df_off: pd.DataFrame,
    preferred_on_routes: set[str],
    lima_quotas: dict[str, int],
) -> None:
    """Cover ON routes in OFF through quota-neutral, strategy-neutral swaps."""
    if not preferred_on_routes:
        return

    route_key = df_off["RUT.COM"].astype(str)
    for preferred_route in sorted(preferred_on_routes):
        if bool((route_key.eq(preferred_route) & df_off[config.SELECTION_COL].eq("T")).any()):
            continue
        incoming_candidates = _rank(
            df_off.loc[
                route_key.eq(preferred_route)
                & df_off[config.SELECTION_COL].eq("NO")
                & df_off["_off_qualified_route"]
            ]
        )
        for incoming_index, incoming in incoming_candidates.iterrows():
            group_mask = (
                df_off["NOMBRE"].eq(incoming["NOMBRE"])
                & df_off["Subcanal2"].eq(incoming["Subcanal2"])
                & df_off["_segment"].eq(incoming["_segment"])
            )
            selected_route_counts = (
                df_off.loc[df_off[config.SELECTION_COL].eq("T"), "RUT.COM"].astype(str).value_counts()
            )
            protected_overlap = route_key.isin(preferred_on_routes) & route_key.map(selected_route_counts).fillna(0).le(1)
            removable = (
                group_mask
                & df_off[config.SELECTION_COL].eq("T")
                & ~df_off["_mandatory"]
                & ~protected_overlap
            )
            if incoming["NOMBRE"] == "LIMA":
                for location, minimum in lima_quotas.items():
                    location_selected = int(
                        (
                            df_off["NOMBRE"].eq("LIMA")
                            & df_off["LOC.COM"].eq(location)
                            & df_off[config.SELECTION_COL].eq("T")
                        ).sum()
                    )
                    if location_selected <= int(minimum):
                        removable &= ~(
                            df_off["NOMBRE"].eq("LIMA") & df_off["LOC.COM"].eq(location)
                        )
            outgoing = df_off.loc[removable].sort_values(
                by=["_preferred_on_route", "_is_titan", "Esparta_Flag", "Ventas", "CODIGO"],
                ascending=[True, True, True, True, False], kind="stable",
            )
            if outgoing.empty:
                continue
            df_off.loc[outgoing.index[0], config.SELECTION_COL] = "NO"
            df_off.loc[incoming_index, config.SELECTION_COL] = "T"
            break


def _route_specialized_targets(group: pd.DataFrame, total_target: int) -> tuple[dict[str, int], int]:
    """Balance specialized SI titles by route inside the 10–30 operating band."""
    specialized = group.loc[group["_is_specialized"]].copy()
    if specialized.empty:
        return {}, 0

    specialized["_route_key"] = specialized["RUT.COM"].astype(str)
    route_stats = specialized.groupby("_route_key", dropna=False).agg(
        available=("CODIGO", "size"), mandatory_si=("_mandatory", "sum")
    ).astype(int)
    mandatory_si = int(route_stats["mandatory_si"].sum())
    mandatory_no = int((group["_mandatory"] & ~group["_is_specialized"]).sum())
    available_no = int((~group["_is_specialized"]).sum())
    effective_total = max(int(total_target), mandatory_si + mandatory_no)
    max_si_within_total = max(0, effective_total - mandatory_no)
    references = {}
    for route, row in route_stats.iterrows():
        available = int(row.available)
        mandatory = int(row.mandatory_si)
        operating_available = available - 1 if available > mandatory else available
        if operating_available >= int(config.ON_SPECIALIZED_ROUTE_ACTIVE_MIN):
            references[str(route)] = max(
                mandatory,
                min(int(config.ON_SPECIALIZED_ROUTE_MIN), operating_available),
            )
        elif mandatory > 0:
            # A short route may only be active when it contains a mandatory
            # Titán/Fénix point. Reserve one candidate whenever possible.
            references[str(route)] = max(mandatory, operating_available)
        else:
            # Do not inflate the specialized target with an optional route
            # that cannot reach the operating minimum of ten.
            references[str(route)] = 0
    ideal_si = sum(references.values())
    target_si = max(
        mandatory_si,
        min(ideal_si, int(route_stats["available"].sum()), max_si_within_total),
    )
    if (
        0 < target_si < int(config.ON_SPECIALIZED_ROUTE_ACTIVE_MIN)
        and mandatory_si == 0
        and available_no >= effective_total
    ):
        target_si = 0
    if target_si <= 0:
        return {str(route): 0 for route in route_stats.index}, 0
    targets = _balanced_on_route_targets(
        specialized,
        target_si,
        reserve_substitute=True,
    )
    return targets, target_si


def _select_on_route_specialized_targets(
    df_on: pd.DataFrame, city_targets: pd.Series, strategy_targets: pd.Series
) -> dict[tuple[str, str], int]:
    """Preselect the allocated SI route targets, prioritizing the historical strategy mix."""
    all_targets: dict[tuple[str, str], int] = {}
    cities = sorted(set(df_on["NOMBRE"].dropna().tolist()))
    for city in cities:
        city_mask = df_on["NOMBRE"].eq(city)
        group = df_on.loc[city_mask]
        total_target = int(city_targets.get((city, "ON"), 0))
        route_targets, _target_si = _route_specialized_targets(group, total_target)
        historical = _strategy_targets_for(city, "ON", strategy_targets)

        for route_key, route_target in sorted(route_targets.items()):
            all_targets[(city, route_key)] = route_target
            route_mask = city_mask & df_on["RUT.COM"].astype(str).eq(route_key)
            while int((route_mask & df_on["_is_specialized"] & df_on[config.SELECTION_COL].eq("T")).sum()) < route_target:
                available = df_on.loc[
                    route_mask & df_on["_is_specialized"] & df_on[config.SELECTION_COL].eq("NO")
                ].copy()
                if available.empty:
                    break
                selected_by_segment = {
                    segment: int((city_mask & df_on["_segment"].eq(segment) & df_on[config.SELECTION_COL].eq("T")).sum())
                    for segment in ("EG", "GR", "ME", "PE")
                }
                desired, _lower, _upper = _strategy_percentage_bands(historical, max(total_target, 1))
                available["_strategy_gap"] = available["_segment"].map(
                    lambda segment: desired.get(segment, 0) - selected_by_segment.get(segment, 0)
                ).fillna(-10_000)
                ranked = _rank(available, prefer_specialized=True).sort_values(
                    by=["_strategy_gap", "_mandatory", "Ventas", "CODIGO"],
                    ascending=[False, False, False, True], kind="stable",
                )
                _set_selected(df_on, [ranked.index[0]])
    return all_targets


def select_canal_on(df_on: pd.DataFrame, city_targets: pd.Series, strategy_targets: pd.Series) -> pd.DataFrame:
    df_on = _prepare(df_on)
    # Rule 1 also applies to ON before any quota completion.
    _set_selected(df_on, df_on.index[df_on["_mandatory"]])
    route_targets = _select_on_route_specialized_targets(df_on, city_targets, strategy_targets)
    active_routes = set(
        df_on.loc[df_on[config.SELECTION_COL].eq("T"), "RUT.COM"].astype(str)
    )
    df_on["_preferred_on_route"] = df_on["RUT.COM"].astype(str).isin(active_routes)
    on_targets = city_targets[city_targets.index.get_level_values("Subcanal2") == "ON"]
    for (city, subchannel), target in on_targets.items():
        non_specialized = ~df_on["_is_specialized"]
        specialized = df_on["_is_specialized"]
        _fill_city_total(
            df_on, city, subchannel, int(target), strategy_targets,
            non_specialized, specialized, False,
        )

    _rebalance_on_specialization(df_on, strategy_targets, route_targets)
    _concentrate_on_non_specialized_titles(df_on, route_targets)
    _ensure_on_route_substitute_capacity(df_on)
    _rebalance_on_strategy_within_routes(df_on, strategy_targets)
    for (city, subchannel), target in on_targets.items():
        actual = int((
            df_on["NOMBRE"].eq(city)
            & df_on["Subcanal2"].eq(subchannel)
            & df_on[config.SELECTION_COL].eq("T")
        ).sum())
        if actual != int(target):
            raise RuntimeError(
                f"La cuota ON de {city} debe ser {int(target):,}, pero quedó en {actual:,}."
            )
    _assign_substitutes(df_on)
    for (city, route), group in df_on.groupby(["NOMBRE", "RUT.COM"], dropna=False):
        if not bool(group[config.SELECTION_COL].eq("T").any()):
            continue
        if bool(group[config.SELECTION_COL].isin(["S1", "S2", "S3"]).any()):
            continue
        optional_titles = group[config.SELECTION_COL].eq("T") & ~group["_mandatory"]
        if bool(optional_titles.any()):
            raise RuntimeError(
                f"La ruta ON {route} de {city} quedó con titulares opcionales pero sin suplentes."
            )
    return df_on.drop(columns=[column for column in df_on.columns if column.startswith("_")], errors="ignore")


def _rebalance_on_specialization(
    df_on: pd.DataFrame,
    strategy_targets: pd.Series,
    route_targets: dict[tuple[str, str], int],
) -> None:
    """Preserve route targets closest to 30; use 90/10 only as a secondary reference."""
    for (city, subchannel), _group in df_on.groupby(["NOMBRE", "Subcanal2"], dropna=False):
        group_mask = df_on["NOMBRE"].eq(city) & df_on["Subcanal2"].eq(subchannel)
        total = int((group_mask & df_on[config.SELECTION_COL].eq("T")).sum())
        target_si = sum(
            int(target) for (target_city, _route), target in route_targets.items()
            if target_city == city
        )
        _desired, lower, upper = _strategy_percentage_bands(
            _strategy_targets_for(city, subchannel, strategy_targets), total
        )

        # Raise SI only when the route-driven target has not yet been met,
        # first within the same strategy and then across strategies.
        while int((group_mask & df_on[config.SELECTION_COL].eq("T") & df_on["_is_specialized"]).sum()) < target_si:
            replacement_found = False
            for segment in ("EG", "GR", "ME", "PE"):
                segment_mask = group_mask & df_on["_segment"].eq(segment)
                outgoing = df_on.loc[
                    segment_mask & df_on[config.SELECTION_COL].eq("T")
                    & ~df_on["_is_specialized"] & ~df_on["_mandatory"]
                ].sort_values(
                    by=["_mandatory", "_is_titan", "Esparta_Flag", "Ventas", "CODIGO"],
                    ascending=[True, True, True, True, False], kind="stable",
                )
                incoming = df_on.loc[
                    segment_mask & df_on[config.SELECTION_COL].eq("NO") & df_on["_is_specialized"]
                ]
                if outgoing.empty or incoming.empty:
                    continue
                df_on.loc[outgoing.index[0], config.SELECTION_COL] = "NO"
                df_on.loc[_rank(incoming, prefer_specialized=True).index[0], config.SELECTION_COL] = "T"
                replacement_found = True
                break
            if replacement_found:
                continue

            selected_by_segment = {
                segment: int((group_mask & df_on["_segment"].eq(segment) & df_on[config.SELECTION_COL].eq("T")).sum())
                for segment in ("EG", "GR", "ME", "PE")
            }
            outgoing = df_on.loc[
                group_mask & df_on[config.SELECTION_COL].eq("T")
                & ~df_on["_is_specialized"] & ~df_on["_mandatory"]
                & df_on["_segment"].map(lambda value: selected_by_segment.get(value, 0) > lower.get(value, 0))
            ].sort_values(
                by=["_mandatory", "_is_titan", "Esparta_Flag", "Ventas", "CODIGO"],
                ascending=[True, True, True, True, False], kind="stable",
            )
            incoming = df_on.loc[
                group_mask & df_on[config.SELECTION_COL].eq("NO") & df_on["_is_specialized"]
                & df_on["_segment"].map(lambda value: selected_by_segment.get(value, 0) < upper.get(value, 0))
            ]
            if outgoing.empty or incoming.empty:
                # Final fallback: the route target has priority when the
                # strategy band cannot provide a feasible swap.
                outgoing = df_on.loc[
                    group_mask & df_on[config.SELECTION_COL].eq("T")
                    & ~df_on["_is_specialized"] & ~df_on["_mandatory"]
                ].sort_values(
                    by=["_mandatory", "_is_titan", "Esparta_Flag", "Ventas", "CODIGO"],
                    ascending=[True, True, True, True, False], kind="stable",
                )
                incoming = df_on.loc[
                    group_mask & df_on[config.SELECTION_COL].eq("NO") & df_on["_is_specialized"]
                ]
                if outgoing.empty or incoming.empty:
                    break
            df_on.loc[outgoing.index[0], config.SELECTION_COL] = "NO"
            df_on.loc[_rank(incoming, prefer_specialized=True).index[0], config.SELECTION_COL] = "T"

        # Remove SI selected beyond the route-driven target without removing
        # mandatory SI or taking a route below its target closest to 30.
        while int((group_mask & df_on[config.SELECTION_COL].eq("T") & df_on["_is_specialized"]).sum()) > target_si:
            selected_si_by_route = (
                df_on.loc[group_mask & df_on[config.SELECTION_COL].eq("T") & df_on["_is_specialized"]]
                .assign(_route_key=lambda frame: frame["RUT.COM"].astype(str))
                .groupby("_route_key").size()
            )
            removable_routes = {
                route for route, count in selected_si_by_route.items()
                if int(count) > int(route_targets.get((city, route), 0))
            }
            removable_route = df_on["RUT.COM"].astype(str).isin(removable_routes)
            replacement_found = False
            for segment in ("EG", "GR", "ME", "PE"):
                segment_mask = group_mask & df_on["_segment"].eq(segment)
                outgoing = df_on.loc[
                    segment_mask & removable_route & df_on[config.SELECTION_COL].eq("T")
                    & df_on["_is_specialized"] & ~df_on["_mandatory"]
                ].sort_values(
                    by=["_mandatory", "_is_titan", "Esparta_Flag", "Ventas", "CODIGO"],
                    ascending=[True, True, True, True, False], kind="stable",
                )
                incoming = df_on.loc[
                    segment_mask & df_on[config.SELECTION_COL].eq("NO") & ~df_on["_is_specialized"]
                ]
                if outgoing.empty or incoming.empty:
                    continue
                df_on.loc[outgoing.index[0], config.SELECTION_COL] = "NO"
                df_on.loc[_rank(incoming, prefer_specialized=False).index[0], config.SELECTION_COL] = "T"
                replacement_found = True
                break
            if replacement_found:
                continue

            selected_by_segment = {
                segment: int((group_mask & df_on["_segment"].eq(segment) & df_on[config.SELECTION_COL].eq("T")).sum())
                for segment in ("EG", "GR", "ME", "PE")
            }
            outgoing = df_on.loc[
                group_mask & removable_route & df_on[config.SELECTION_COL].eq("T")
                & df_on["_is_specialized"] & ~df_on["_mandatory"]
                & df_on["_segment"].map(lambda value: selected_by_segment.get(value, 0) > lower.get(value, 0))
            ].sort_values(
                by=["_mandatory", "_is_titan", "Esparta_Flag", "Ventas", "CODIGO"],
                ascending=[True, True, True, True, False], kind="stable",
            )
            incoming = df_on.loc[
                group_mask & df_on[config.SELECTION_COL].eq("NO") & ~df_on["_is_specialized"]
                & df_on["_segment"].map(lambda value: selected_by_segment.get(value, 0) < upper.get(value, 0))
            ]
            if outgoing.empty or incoming.empty:
                # Final fallback for the closest possible route result. Route
                # targets and mandatory points remain protected.
                outgoing = df_on.loc[
                    group_mask & removable_route & df_on[config.SELECTION_COL].eq("T")
                    & df_on["_is_specialized"] & ~df_on["_mandatory"]
                ].sort_values(
                    by=["_mandatory", "_is_titan", "Esparta_Flag", "Ventas", "CODIGO"],
                    ascending=[True, True, True, True, False], kind="stable",
                )
                incoming = df_on.loc[
                    group_mask & df_on[config.SELECTION_COL].eq("NO") & ~df_on["_is_specialized"]
                ]
                if outgoing.empty or incoming.empty:
                    break
            df_on.loc[outgoing.index[0], config.SELECTION_COL] = "NO"
            df_on.loc[_rank(incoming, prefer_specialized=False).index[0], config.SELECTION_COL] = "T"


def _concentrate_on_non_specialized_titles(
    df_on: pd.DataFrame,
    route_targets: dict[tuple[str, str], int] | None = None,
) -> None:
    """Pack optional ON/NO titles into viable route groups instead of one per route.

    Mandatory Titán/Fénix points are never moved. Optional non-specialized
    titles are rebuilt by city and subchannel, opening a route only when it can
    hold at least ``ON_NON_SPECIALIZED_ROUTE_MIN`` titles. Any residue that
    cannot form a viable route is replaced with specialized points, preferably
    within the same strategy, so the city total and strategy mix stay intact.
    """
    minimum = int(config.ON_NON_SPECIALIZED_ROUTE_MIN)
    route_targets = route_targets or {}
    route_key = df_on["RUT.COM"].astype(str)

    for (city, subchannel), _group in df_on.groupby(["NOMBRE", "Subcanal2"], dropna=False):
        group_mask = df_on["NOMBRE"].eq(city) & df_on["Subcanal2"].eq(subchannel)
        strategy_before = Counter(
            df_on.loc[
                group_mask & df_on[config.SELECTION_COL].eq("T"), "_segment"
            ].astype(str)
        )
        non_specialized = group_mask & ~df_on["_is_specialized"]
        optional_selected = (
            non_specialized
            & df_on[config.SELECTION_COL].eq("T")
            & ~df_on["_mandatory"]
        )
        if not bool(optional_selected.any()):
            continue

        optional_targets = (
            df_on.loc[optional_selected, "_segment"].value_counts().astype(int).to_dict()
        )
        df_on.loc[optional_selected, config.SELECTION_COL] = "NO"
        remaining_by_segment = Counter(optional_targets)
        remaining_optional = int(sum(remaining_by_segment.values()))

        def specialized_capacity_mask() -> pd.Series:
            selected_si_counts = (
                df_on.loc[
                    group_mask
                    & df_on[config.SELECTION_COL].eq("T")
                    & df_on["_is_specialized"]
                ]
                .assign(_route_key=lambda frame: frame["RUT.COM"].astype(str))
                .groupby("_route_key").size()
            )
            return route_key.map(
                lambda route: int(selected_si_counts.get(str(route), 0))
                < int(route_targets.get((city, str(route)), 0))
            )

        # Complete mandatory NO routes first when they can reach the operating
        # minimum; afterwards open the fewest high-capacity routes possible.
        while remaining_optional > 0:
            pending = (
                non_specialized
                & df_on[config.SELECTION_COL].eq("NO")
                & df_on["_segment"].map(lambda value: remaining_by_segment.get(value, 0) > 0)
            )
            if not bool(pending.any()):
                break

            route_options = []
            for route, candidates in df_on.loc[pending].groupby(route_key[pending], dropna=False):
                capacity_by_segment = candidates["_segment"].value_counts().to_dict()
                coverage = int(sum(
                    min(int(capacity_by_segment.get(segment, 0)), int(needed))
                    for segment, needed in remaining_by_segment.items()
                    if needed > 0
                ))
                already_selected_no = int((
                    group_mask
                    & route_key.eq(str(route))
                    & ~df_on["_is_specialized"]
                    & df_on[config.SELECTION_COL].eq("T")
                ).sum())
                required_to_activate = max(0, minimum - already_selected_no)
                can_activate = coverage >= required_to_activate and (
                    already_selected_no > 0 or coverage >= minimum
                )
                if not can_activate or coverage <= 0:
                    continue
                has_any_title = bool((
                    group_mask
                    & route_key.eq(str(route))
                    & df_on[config.SELECTION_COL].eq("T")
                ).any())
                route_options.append((
                    str(route), coverage, already_selected_no, has_any_title,
                    float(pd.to_numeric(candidates["Ventas"], errors="coerce").fillna(0).sum()),
                ))

            if not route_options:
                break
            chosen_route = max(
                route_options,
                key=lambda value: (
                    value[2] > 0, value[1], value[3], value[4], value[0]
                ),
            )[0]
            chosen_before = remaining_optional
            for segment in sorted(
                remaining_by_segment,
                key=lambda value: remaining_by_segment[value],
                reverse=True,
            ):
                needed = int(remaining_by_segment[segment])
                if needed <= 0:
                    continue
                candidates = df_on.loc[
                    non_specialized
                    & route_key.eq(chosen_route)
                    & df_on["_segment"].eq(segment)
                    & df_on[config.SELECTION_COL].eq("NO")
                ]
                chosen = _rank(candidates).index[:needed]
                _set_selected(df_on, chosen)
                taken = len(chosen)
                remaining_by_segment[segment] -= taken
                remaining_optional -= taken
            if remaining_optional == chosen_before:
                break

        # A residue of 1-3 optional NO points would create another one-point
        # route. Replace it with SI points, first in the same strategy and on an
        # already active route. This keeps the title total unchanged.
        for segment in list(remaining_by_segment):
            needed = int(remaining_by_segment[segment])
            if needed <= 0:
                continue
            incoming = df_on.loc[
                group_mask
                & df_on["_is_specialized"]
                & df_on["_segment"].eq(segment)
                & df_on[config.SELECTION_COL].eq("NO")
                & specialized_capacity_mask()
            ]
            chosen = _rank(incoming, prefer_specialized=True).index[:needed]
            _set_selected(df_on, chosen)
            taken = len(chosen)
            remaining_by_segment[segment] -= taken
            remaining_optional -= taken

        # Preserve the exact city total if the same-strategy SI universe was
        # insufficient. Cross-strategy SI is preferred to opening a tiny NO
        # route; a final NO fallback is used only when SI is exhausted.
        if remaining_optional > 0:
            incoming_si = df_on.loc[
                group_mask
                & df_on["_is_specialized"]
                & df_on[config.SELECTION_COL].eq("NO")
                & specialized_capacity_mask()
            ]
            chosen = _rank(incoming_si, prefer_specialized=True).index[:remaining_optional]
            _set_selected(df_on, chosen)
            remaining_optional -= len(chosen)

        # If SI is exhausted, use spare capacity in an already active NO route
        # before considering a new route. The strategy is secondary to avoiding
        # another isolated operational stop.
        if remaining_optional > 0:
            selected_no_counts = (
                df_on.loc[non_specialized & df_on[config.SELECTION_COL].eq("T")]
                .assign(_route_key=lambda frame: frame["RUT.COM"].astype(str))
                .groupby("_route_key").size()
            )
            active_no_routes = set(
                str(route) for route, count in selected_no_counts.items()
                if int(count) >= minimum
            )
            incoming_active_no = df_on.loc[
                non_specialized
                & route_key.isin(active_no_routes)
                & df_on[config.SELECTION_COL].eq("NO")
            ]
            chosen = _rank(incoming_active_no).index[:remaining_optional]
            _set_selected(df_on, chosen)
            remaining_optional -= len(chosen)

        # A final city residue may be smaller than four. When a fresh NO route
        # has enough capacity, open it with a complete group and remove the
        # equivalent excess from optional SI titles. This preserves the exact
        # city total while respecting the route minimum.
        while remaining_optional > 0:
            pending_no = df_on.loc[
                non_specialized & df_on[config.SELECTION_COL].eq("NO")
            ]
            if pending_no.empty:
                break
            selected_no_counts = (
                df_on.loc[non_specialized & df_on[config.SELECTION_COL].eq("T")]
                .assign(_route_key=lambda frame: frame["RUT.COM"].astype(str))
                .groupby("_route_key").size()
            )
            route_groups = [
                (
                    str(route), candidates,
                    max(1, minimum - int(selected_no_counts.get(str(route), 0))),
                )
                for route, candidates in pending_no.groupby(route_key.loc[pending_no.index], dropna=False)
                if int(selected_no_counts.get(str(route), 0)) < minimum
                and len(candidates) >= max(1, minimum - int(selected_no_counts.get(str(route), 0)))
            ]
            if not route_groups:
                break
            selected_si_counts = (
                df_on.loc[
                    group_mask
                    & df_on["_is_specialized"]
                    & df_on[config.SELECTION_COL].eq("T")
                ]
                .assign(_route_key=lambda frame: frame["RUT.COM"].astype(str))
                .groupby("_route_key").size()
            )
            above_specialized_target = route_key.map(
                lambda route: int(selected_si_counts.get(str(route), 0))
                > int(route_targets.get((city, str(route)), 0))
            )
            removable_si = df_on.loc[
                group_mask
                & df_on["_is_specialized"]
                & df_on[config.SELECTION_COL].eq("T")
                & ~df_on["_mandatory"]
                & above_specialized_target
            ].sort_values(
                by=["_preferred_on_route", "Esparta_Flag", "Ventas", "CODIGO"],
                ascending=[True, True, True, False],
                kind="stable",
            )
            max_group_size = remaining_optional + len(removable_si)
            viable = [
                (route, candidates, minimum_addition)
                for route, candidates, minimum_addition in route_groups
                if min(len(candidates), max_group_size) >= minimum_addition
            ]
            if not viable:
                break
            chosen_route, route_candidates, minimum_addition = max(
                viable,
                key=lambda item: (
                    min(len(item[1]), max_group_size),
                    float(pd.to_numeric(item[1]["Ventas"], errors="coerce").fillna(0).sum()),
                    item[0],
                ),
            )
            group_size = min(
                len(route_candidates), max(minimum_addition, remaining_optional), max_group_size
            )
            chosen = _rank(route_candidates).index[:group_size]
            _set_selected(df_on, chosen)
            excess = max(0, len(chosen) - remaining_optional)
            if excess > 0:
                df_on.loc[removable_si.index[:excess], config.SELECTION_COL] = "NO"
            remaining_optional = max(0, remaining_optional - len(chosen) + excess)

        if remaining_optional > 0:
            incoming_no = df_on.loc[
                non_specialized & df_on[config.SELECTION_COL].eq("NO")
            ]
            chosen = _rank(incoming_no).index[:remaining_optional]
            _set_selected(df_on, chosen)

        # Last consolidation pass: if the exact quota forced optional titles
        # into a route below four, move the whole optional block to spare places
        # on an already viable NO route whenever such capacity exists.
        while True:
            selected_no = df_on.loc[
                non_specialized & df_on[config.SELECTION_COL].eq("T")
            ].copy()
            if selected_no.empty:
                break
            selected_no["_route_key_local"] = selected_no["RUT.COM"].astype(str)
            selected_counts = selected_no.groupby("_route_key_local").size()
            optional_counts = (
                selected_no.loc[~selected_no["_mandatory"]]
                .groupby("_route_key_local").size()
            )
            small_routes = [
                str(route) for route, count in selected_counts.items()
                if int(count) < minimum and int(optional_counts.get(route, 0)) > 0
            ]
            moved = False
            for small_route in small_routes:
                outgoing = _rank(df_on.loc[
                    non_specialized
                    & route_key.eq(small_route)
                    & df_on[config.SELECTION_COL].eq("T")
                    & ~df_on["_mandatory"]
                ])
                if outgoing.empty:
                    continue
                viable_routes = {
                    str(route) for route, count in selected_counts.items()
                    if int(count) >= minimum and str(route) != small_route
                }
                incoming_pool = df_on.loc[
                    non_specialized
                    & route_key.isin(viable_routes)
                    & df_on[config.SELECTION_COL].eq("NO")
                ]
                if len(incoming_pool) < len(outgoing):
                    continue
                chosen_incoming = []
                remaining_pool = incoming_pool.copy()
                for _index, outgoing_row in outgoing.iterrows():
                    same_segment = remaining_pool.loc[
                        remaining_pool["_segment"].eq(outgoing_row["_segment"])
                    ]
                    candidate_pool = same_segment if not same_segment.empty else remaining_pool
                    if candidate_pool.empty:
                        break
                    chosen_index = _rank(candidate_pool).index[0]
                    chosen_incoming.append(chosen_index)
                    remaining_pool = remaining_pool.drop(index=chosen_index)
                if len(chosen_incoming) != len(outgoing):
                    continue
                df_on.loc[outgoing.index, config.SELECTION_COL] = "NO"
                _set_selected(df_on, chosen_incoming)
                moved = True
                break
            if not moved:
                break

        _restore_on_group_strategy_counts(df_on, group_mask, strategy_before)


def _restore_on_group_strategy_counts(
    df_on: pd.DataFrame,
    group_mask: pd.Series,
    target_counts: Counter,
) -> None:
    """Undo strategy drift introduced while grouping non-specialized routes."""
    route_key = df_on["RUT.COM"].astype(str)
    non_specialized = group_mask & ~df_on["_is_specialized"]
    segments = ("EG", "GR", "ME", "PE")

    for _ in range(int(sum(target_counts.values())) + 1):
        current = Counter(
            df_on.loc[
                group_mask & df_on[config.SELECTION_COL].eq("T"), "_segment"
            ].astype(str)
        )
        deficits = [
            segment for segment in segments
            if current.get(segment, 0) < int(target_counts.get(segment, 0))
        ]
        donors = [
            segment for segment in segments
            if current.get(segment, 0) > int(target_counts.get(segment, 0))
        ]
        if not deficits or not donors:
            return

        swapped = False
        for incoming_segment in deficits:
            incoming_pool = df_on.loc[
                non_specialized
                & df_on["_segment"].eq(incoming_segment)
                & df_on[config.SELECTION_COL].eq("NO")
            ].copy()
            if incoming_pool.empty:
                continue
            for outgoing_segment in donors:
                outgoing_pool = df_on.loc[
                    non_specialized
                    & df_on["_segment"].eq(outgoing_segment)
                    & df_on[config.SELECTION_COL].eq("T")
                    & ~df_on["_mandatory"]
                ].copy()
                if outgoing_pool.empty:
                    continue

                # A swap inside one route preserves every route count. Prefer
                # it before moving a title between two non-specialized routes.
                common_routes = sorted(
                    set(route_key.loc[incoming_pool.index])
                    & set(route_key.loc[outgoing_pool.index])
                )
                if common_routes:
                    route = common_routes[0]
                    incoming = _rank(
                        incoming_pool.loc[route_key.loc[incoming_pool.index].eq(route)]
                    ).index[0]
                    outgoing = _rank(
                        outgoing_pool.loc[route_key.loc[outgoing_pool.index].eq(route)]
                    ).index[-1]
                else:
                    # Across routes, prefer adding to an already active route
                    # that will still retain at least one substitute candidate.
                    title_counts = (
                        df_on.loc[non_specialized & df_on[config.SELECTION_COL].eq("T")]
                        .assign(_route_key_local=lambda frame: frame["RUT.COM"].astype(str))
                        .groupby("_route_key_local").size()
                    )
                    remaining_counts = (
                        df_on.loc[group_mask & df_on[config.SELECTION_COL].eq("NO")]
                        .assign(_route_key_local=lambda frame: frame["RUT.COM"].astype(str))
                        .groupby("_route_key_local").size()
                    )
                    incoming_pool["_active_route"] = route_key.loc[incoming_pool.index].map(
                        lambda route: int(title_counts.get(str(route), 0)) > 0
                    )
                    incoming_pool["_keeps_substitute"] = route_key.loc[incoming_pool.index].map(
                        lambda route: int(remaining_counts.get(str(route), 0)) >= 2
                    )
                    viable_incoming = incoming_pool.loc[incoming_pool["_keeps_substitute"]]
                    if viable_incoming.empty:
                        continue
                    ranked_incoming = _rank(viable_incoming).sort_values(
                        by=["_active_route", "_keeps_substitute"],
                        ascending=[False, False],
                        kind="stable",
                    )
                    incoming = ranked_incoming.index[0]
                    outgoing_pool["_route_titles"] = route_key.loc[outgoing_pool.index].map(
                        lambda route: int(title_counts.get(str(route), 0))
                    )
                    outgoing = _rank(outgoing_pool).sort_values(
                        by="_route_titles", ascending=True, kind="stable"
                    ).index[-1]

                df_on.loc[outgoing, config.SELECTION_COL] = "NO"
                df_on.loc[incoming, config.SELECTION_COL] = "T"
                swapped = True
                break
            if swapped:
                break
        if not swapped:
            return


def _rebalance_on_strategy_within_routes(
    df_on: pd.DataFrame,
    strategy_targets: pd.Series,
) -> None:
    """Repair feasible +/-1 pp strategy gaps without changing route totals."""
    segments = ("EG", "GR", "ME", "PE")
    route_key = df_on["RUT.COM"].astype(str)
    for (city, subchannel), _group in df_on.groupby(["NOMBRE", "Subcanal2"], dropna=False):
        group_mask = df_on["NOMBRE"].eq(city) & df_on["Subcanal2"].eq(subchannel)
        total = int((group_mask & df_on[config.SELECTION_COL].eq("T")).sum())
        if total <= 0:
            continue
        historical = _strategy_targets_for(city, subchannel, strategy_targets)
        _desired, lower, upper = _strategy_percentage_bands(historical, total)

        for _ in range(total + 1):
            current = Counter(
                df_on.loc[
                    group_mask & df_on[config.SELECTION_COL].eq("T"), "_segment"
                ].astype(str)
            )
            deficits = [segment for segment in segments if current.get(segment, 0) < lower.get(segment, 0)]
            donors = [segment for segment in segments if current.get(segment, 0) > lower.get(segment, 0)]
            if not deficits or not donors:
                break

            swapped = False
            for incoming_segment in deficits:
                for outgoing_segment in donors:
                    for specialized in (True, False):
                        specialization = df_on["_is_specialized"].eq(specialized)
                        outgoing_pool = df_on.loc[
                            group_mask
                            & specialization
                            & df_on["_segment"].eq(outgoing_segment)
                            & df_on[config.SELECTION_COL].eq("T")
                            & ~df_on["_mandatory"]
                        ]
                        incoming_pool = df_on.loc[
                            group_mask
                            & specialization
                            & df_on["_segment"].eq(incoming_segment)
                            & df_on[config.SELECTION_COL].eq("NO")
                        ]
                        common_routes = sorted(
                            set(route_key.loc[outgoing_pool.index])
                            & set(route_key.loc[incoming_pool.index])
                        )
                        if not common_routes:
                            continue
                        route = common_routes[0]
                        outgoing = _rank(
                            outgoing_pool.loc[route_key.loc[outgoing_pool.index].eq(route)]
                        ).index[-1]
                        incoming = _rank(
                            incoming_pool.loc[route_key.loc[incoming_pool.index].eq(route)],
                            prefer_specialized=specialized,
                        ).index[0]
                        df_on.loc[outgoing, config.SELECTION_COL] = "NO"
                        df_on.loc[incoming, config.SELECTION_COL] = "T"
                        swapped = True
                        break
                    if swapped:
                        break
                if swapped:
                    break
            if not swapped:
                break


def _ensure_on_route_substitute_capacity(df_on: pd.DataFrame) -> None:
    """Move optional titles away from routes that would have no substitute."""
    route_key = df_on["RUT.COM"].astype(str)
    for _ in range(len(df_on) + 1):
        full_route = None
        for (city, route), group in df_on.groupby(["NOMBRE", "RUT.COM"], dropna=False):
            title_mask = group[config.SELECTION_COL].eq("T")
            if bool(title_mask.any()) and bool(title_mask.all()) and bool((title_mask & ~group["_mandatory"]).any()):
                full_route = (city, str(route), group)
                break
        if full_route is None:
            return

        city, route, group = full_route
        outgoing_pool = group.loc[
            group[config.SELECTION_COL].eq("T") & ~group["_mandatory"]
        ].copy()
        if outgoing_pool.empty:
            return
        outgoing = _rank(outgoing_pool).index[-1]
        outgoing_segment = str(df_on.at[outgoing, "_segment"])
        outgoing_specialized = bool(df_on.at[outgoing, "_is_specialized"])

        city_mask = df_on["NOMBRE"].eq(city)
        candidates = df_on.loc[
            city_mask
            & df_on[config.SELECTION_COL].eq("NO")
            & ~route_key.eq(route)
        ].copy()
        if candidates.empty:
            return
        remaining_by_route = (
            df_on.loc[city_mask & df_on[config.SELECTION_COL].eq("NO")]
            .assign(_route_key_local=lambda frame: frame["RUT.COM"].astype(str))
            .groupby("_route_key_local").size()
        )
        candidates = candidates.loc[
            route_key.loc[candidates.index].map(
                lambda candidate_route: int(remaining_by_route.get(str(candidate_route), 0)) >= 2
            )
        ].copy()
        if candidates.empty:
            return

        same_segment = candidates.loc[candidates["_segment"].eq(outgoing_segment)]
        candidates = same_segment if not same_segment.empty else candidates
        title_routes = set(
            route_key.loc[
                city_mask & df_on[config.SELECTION_COL].eq("T")
            ]
        )
        candidates["_same_specialization"] = candidates["_is_specialized"].eq(outgoing_specialized)
        candidates["_active_route"] = route_key.loc[candidates.index].isin(title_routes)
        incoming = _rank(candidates, prefer_specialized=outgoing_specialized).sort_values(
            by=["_same_specialization", "_active_route"],
            ascending=[False, False],
            kind="stable",
        ).index[0]

        df_on.loc[outgoing, config.SELECTION_COL] = "NO"
        df_on.loc[incoming, config.SELECTION_COL] = "T"


def _balanced_on_route_targets(
    group: pd.DataFrame,
    total_target: int,
    reserve_substitute: bool = True,
) -> dict[str, int]:
    """Return the closest feasible, level 10–30 distribution for active ON routes."""
    route_min = int(config.ON_SPECIALIZED_ROUTE_ACTIVE_MIN)
    route_max = int(config.ON_SPECIALIZED_ROUTE_MAX)
    local = group.assign(_route_key_local=group["RUT.COM"].astype(str))
    stats = local.groupby("_route_key_local", dropna=False).agg(
        available=("CODIGO", "size"),
        mandatory=("_mandatory", "sum"),
        current=(config.SELECTION_COL, lambda values: int(values.eq("T").sum())),
        non_specialized=("_is_specialized", lambda values: int((~values).sum())),
        sales=("Ventas", lambda values: float(pd.to_numeric(values, errors="coerce").fillna(0).sum())),
    )

    lower_if_active: dict[str, int] = {}
    upper: dict[str, int] = {}
    absolute_upper: dict[str, int] = {}
    forced: set[str] = set()
    active: set[str] = set()
    for route, row in stats.iterrows():
        route = str(route)
        available = int(row.available)
        mandatory = int(row.mandatory)
        if available >= route_min:
            floor = max(route_min, mandatory)
            # Reserve one non-title for substitutes whenever this does not make
            # the operating minimum impossible.
            reserving_upper = available - 1
            available_upper = (
                reserving_upper
                if reserve_substitute and reserving_upper >= floor
                else available
            )
            ceiling = min(route_max, available_upper)
        else:
            floor = mandatory
            ceiling = available
        lower_if_active[route] = floor
        upper[route] = max(mandatory, ceiling)
        reserve_if_possible = (
            available - 1
            if reserve_substitute and available - 1 >= mandatory
            else available
        )
        absolute_upper[route] = max(mandatory, reserve_if_possible)
        if mandatory > 0:
            forced.add(route)
            active.add(route)
        elif int(row.current) > 0 and available >= route_min:
            active.add(route)

    def floor_total(routes: set[str]) -> int:
        return sum(lower_if_active[route] for route in routes)

    def upper_total(routes: set[str]) -> int:
        return sum(upper[route] for route in routes)

    # If prior optional routes are too numerous for the minimum of 10, close
    # the smallest ones first. Mandatory routes are never closed.
    removable = sorted(
        active - forced,
        key=lambda route: (
            int(stats.loc[route, "current"]),
            upper[route],
            int(stats.loc[route, "non_specialized"]),
            route,
        ),
    )
    while floor_total(active) > int(total_target) and removable:
        active.remove(removable.pop(0))

    # A small mandatory route may be mathematically unable to reach 10. Relax
    # only its operating floor, never the mandatory count itself.
    if floor_total(active) > int(total_target):
        for route in forced:
            lower_if_active[route] = int(stats.loc[route, "mandatory"])
    if floor_total(active) > int(total_target):
        raise ValueError("Los mínimos de rutas ON superan la cuota disponible del Nombre.")

    # Open extra routes only when the existing active routes cannot hold the
    # city quota under the maximum of 30. Prefer routes rich in NO-specialized
    # points so displaced titles above 30 are absorbed there.
    candidates = sorted(
        [
            route for route in stats.index.astype(str)
            if route not in active and upper[route] >= route_min
        ],
        key=lambda route: (
            int(stats.loc[route, "non_specialized"]),
            upper[route],
            float(stats.loc[route, "sales"]),
            route,
        ),
        reverse=True,
    )
    while upper_total(active) < int(total_target):
        candidate = next(
            (
                route for route in candidates
                if floor_total(active) + lower_if_active[route] <= int(total_target)
            ),
            None,
        )
        if candidate is None:
            # The city quota has explicit priority. If no additional route can
            # reach 10, retain the smallest unavoidable overflow above 30 in an
            # existing route, reserving a substitute whenever capacity permits.
            relaxed = [route for route in active if absolute_upper[route] > upper[route]]
            if relaxed:
                for route in relaxed:
                    upper[route] = absolute_upper[route]
                continue
            # Last mathematical fallback: use a route whose entire universe is
            # below 10. This remains visible as an exception in Controls.
            tiny = [
                route for route in stats.index.astype(str)
                if route not in active and upper[route] > 0
            ]
            if tiny:
                route = max(
                    tiny,
                    key=lambda value: (
                        upper[value], int(stats.loc[value, "non_specialized"]),
                        float(stats.loc[value, "sales"]), value,
                    ),
                )
                lower_if_active[route] = 0
                active.add(route)
                continue
            raise ValueError(
                f"La cuota ON de {int(total_target):,} supera el universo disponible de sus rutas."
            )
        active.add(candidate)
        candidates.remove(candidate)

    targets = {route: 0 for route in stats.index.astype(str)}
    for route in active:
        targets[route] = lower_if_active[route]
    remaining = int(total_target) - sum(targets.values())
    while remaining > 0:
        choices = [route for route in active if targets[route] < upper[route]]
        if not choices:
            raise ValueError("No fue posible completar exactamente la cuota ON dentro del rango 10–30.")
        route = min(
            choices,
            key=lambda value: (
                targets[value],
                -upper[value],
                -int(stats.loc[value, "non_specialized"]),
                value,
            ),
        )
        targets[route] += 1
        remaining -= 1
    return targets


def _assign_substitutes(df: pd.DataFrame) -> None:
    """Assign continuous S1→S3 levels independently inside every title route.

    The function is called separately for OFF and ON, so a substitute can only
    belong to a route with a titular in that same channel. Candidates are also
    split by specialization: a route with NO-specialized titulares receives its
    own S1 first whenever another NO-specialized candidate exists.
    """
    route_columns = ["NOMBRE", "RUT.COM"]
    title_routes = set(
        tuple(str(value) for value in row)
        for row in df.loc[df[config.SELECTION_COL].eq("T"), route_columns].itertuples(index=False, name=None)
    )
    route_pairs = pd.Series(
        [tuple(str(value) for value in row) for row in df[route_columns].itertuples(index=False, name=None)],
        index=df.index,
    )
    candidate_mask = df[config.SELECTION_COL].eq("NO") & route_pairs.isin(title_routes)
    if not bool(candidate_mask.any()):
        return

    df.loc[candidate_mask, "_sub_specialization"] = df.loc[candidate_mask, "_is_specialized"].map(
        {False: "NO", True: "SI"}
    )
    esparta = df["Esparta_Flag"].eq(1) | df["NOMBRE"].isin(config.ESPARTA_CITIES)
    df.loc[candidate_mask, "_sub_priority"] = np.select(
        [
            esparta[candidate_mask] & df.loc[candidate_mask, "_segment"].isin(["ME", "PE"]),
            esparta[candidate_mask],
            df.loc[candidate_mask, "_segment"].isin(["EG", "GR"]),
        ],
        [0, 1, 2],
        default=3,
    )

    levels = ("S1", "S2", "S3")
    grouping = ["NOMBRE", "RUT.COM", "_sub_specialization"]
    for _key, candidates in df.loc[candidate_mask].groupby(grouping, dropna=False, sort=True):
        ordered = candidates.sort_values(
            by=["_sub_priority", "_mandatory", "_prev_t", "Ventas", "CODIGO"],
            ascending=[True, False, True, False, True],
            kind="stable",
        )
        count = len(ordered)
        base, remainder = divmod(count, len(levels))
        start = 0
        for position, level in enumerate(levels):
            level_count = base + (1 if position < remainder else 0)
            if level_count <= 0:
                continue
            indexes = ordered.index[start:start + level_count]
            df.loc[indexes, config.SELECTION_COL] = level
            start += level_count


def build_controls(
    final: pd.DataFrame,
    city_targets: pd.Series,
    historical_city_targets: pd.Series,
    strategy_targets: pd.Series,
    lima_quotas: dict[str, int],
    total_target: int,
    increase_scope: str,
) -> pd.DataFrame:
    rows = []
    selected = final[final[config.SELECTION_COL].eq("T")].copy()
    selected["ESTRATEGICA"] = selected["ESTRATEGICA"].map(_segment)
    actual_total = len(selected)
    rows.append({
        "Control": "Muestra total solicitada", "Canal": "TODOS", "NOMBRE": "VARIOS", "Subcanal2": "",
        "ESTRATEGICA": "", "LOC.COM": "", "Objetivo": int(total_target), "Titulares": actual_total,
        "Diferencia": actual_total - int(total_target), "Estado": "OK" if actual_total == int(total_target) else "REVISAR",
        "Motivo": "Muestra total cumplida" if actual_total == int(total_target) else "Los mínimos obligatorios o la disponibilidad impiden igualar la muestra",
    })
    effective_historical = _anchored_historical_targets(
        final, historical_city_targets, lima_quotas
    )
    historical_on = int(sum(
        value for (_city, subchannel), value in effective_historical.items() if subchannel == "ON"
    ))
    historical_off = int(effective_historical.sum()) - historical_on
    target_on = int(sum(value for (city, subchannel), value in city_targets.items() if subchannel == "ON"))
    target_off = int(city_targets.sum()) - target_on
    fixed_ok = (
        increase_scope == "AMBOS"
        or increase_scope == "OFF" and target_on == historical_on
        or increase_scope == "ON" and target_off == historical_off
    )
    baseline_total = historical_on + historical_off
    rows.append({
        "Control": "Canal que recibe el cambio de muestra", "Canal": increase_scope,
        "NOMBRE": "VARIOS", "Subcanal2": "", "ESTRATEGICA": "", "LOC.COM": "",
        "Objetivo": int(total_target) - baseline_total,
        "Titulares": int(total_target) - baseline_total, "Diferencia": 0,
        "Historico_ON": historical_on, "Objetivo_ON": target_on,
        "Historico_OFF": historical_off, "Objetivo_OFF": target_off,
        "Estado": "OK" if fixed_ok else "REVISAR",
        "Motivo": (
            "El aumento se distribuyó proporcionalmente entre OFF y ON, después de fijar cada cuota histórica"
            if increase_scope == "AMBOS"
            else f"El canal {'ON' if increase_scope == 'OFF' else 'OFF'} permaneció en su base del mes anterior"
            if fixed_ok
            else "Los puntos obligatorios o la disponibilidad impidieron mantener idéntico el canal fijo"
        ),
    })

    actual_city = selected.groupby(CITY_GROUP).size()
    all_city_groups = set(city_targets.index.to_list()) | set(actual_city.index.to_list())
    for city, subchannel in sorted(all_city_groups):
        target = int(city_targets.get((city, subchannel), 0))
        actual = int(actual_city.get((city, subchannel), 0))
        status = "OK" if actual == target else "REVISAR"
        reason = "Cuota proporcional de la nueva muestra cumplida" if status == "OK" else "PDV obligatorios o universo disponible impiden igualar el total"
        rows.append({
            "Control": "Total objetivo por nombre y subcanal", "Canal": "ON" if subchannel == "ON" else "OFF",
            "NOMBRE": city, "Subcanal2": subchannel, "ESTRATEGICA": "", "LOC.COM": "",
            "Objetivo": target, "Titulares": actual, "Diferencia": actual - target, "Estado": status, "Motivo": reason,
        })

    actual_strategy = selected.groupby(STRATEGY_GROUP).size()
    all_strategy_groups = set(strategy_targets.index.to_list()) | set(actual_strategy.index.to_list())
    for city, subchannel, segment in sorted(all_strategy_groups):
        historical_count = int(strategy_targets.get((city, subchannel, segment), 0))
        actual = int(actual_strategy.get((city, subchannel, segment), 0))
        historical_total = int(historical_city_targets.get((city, subchannel), 0))
        current_total = int(actual_city.get((city, subchannel), 0))
        historical_pct = 0 if historical_total == 0 else historical_count / historical_total
        current_pct = 0 if current_total == 0 else actual / current_total
        variation_pp = (current_pct - historical_pct) * 100
        within_band = current_total == 0 or abs(variation_pp) <= 1.000001
        desired, _lower, _upper = _strategy_percentage_bands(
            _strategy_targets_for(city, subchannel, strategy_targets), current_total
        )
        target = int(desired.get(segment, 0))
        rows.append({
            "Control": "Variación por estrategia (máximo +/- 1 pp)", "Canal": "ON" if subchannel == "ON" else "OFF",
            "NOMBRE": city, "Subcanal2": subchannel, "ESTRATEGICA": segment, "LOC.COM": "",
            "Objetivo": target, "Titulares": actual, "Diferencia": actual - target,
            "Historico_%": round(historical_pct * 100, 2), "Actual_%": round(current_pct * 100, 2),
            "Variacion_pp": round(variation_pp, 2),
            "Estado": "OK" if within_band else "REVISAR",
            "Motivo": (
                "Sin cuota actual por falta de universo elegible"
                if current_total == 0
                else "Dentro de +/- 1 punto porcentual"
                if within_band
                else "Obligatorios o universo disponible impiden respetar el límite de +/- 1 pp"
            ),
        })

    # The former 50/50 EG+GR reference is intentionally disabled: preserving
    # each historical strategy percentage within +/- 1 pp is the active rule.
    for city, subchannel in []:
        group = selected["NOMBRE"].eq(city) & selected["Subcanal2"].eq(subchannel)
        total = int(group.sum())
        eg_gr = int((group & selected["ESTRATEGICA"].isin(["EG", "GR"])).sum())
        target_eg_gr = int(round(total * 0.50))
        rows.append({
            "Control": "Mix EG+GR (referencia 50%)", "Canal": "ON" if subchannel == "ON" else "OFF",
            "NOMBRE": city, "Subcanal2": subchannel, "ESTRATEGICA": "EG+GR", "LOC.COM": "",
            "Objetivo": target_eg_gr, "Titulares": eg_gr, "Diferencia": eg_gr - target_eg_gr,
            "Estado": "OK" if abs(eg_gr - target_eg_gr) <= max(1, round(total * 0.05)) else "VARIACION",
            "Motivo": "Meta cercana a 50/50; el total histórico tiene prioridad",
        })

    for location, minimum in lima_quotas.items():
        actual = int((selected["NOMBRE"].eq("LIMA") & selected["Subcanal2"].ne("ON") & selected["LOC.COM"].eq(location)).sum())
        rows.append({
            "Control": "Mínimo Lima por LOC.COM", "Canal": "OFF", "NOMBRE": "LIMA", "Subcanal2": "OFF",
            "ESTRATEGICA": "", "LOC.COM": location, "Objetivo": int(minimum), "Titulares": actual,
            "Diferencia": actual - int(minimum), "Estado": "OK" if actual >= int(minimum) else "REVISAR",
            "Motivo": "Mínimo cumplido" if actual >= int(minimum) else "No hay suficientes elegibles para la cuota mínima",
        })

    on_all = final[final["_CANAL_SELECCION"].eq("ON")].copy()
    on_all["_is_specialized"] = on_all["ESPECIALIZADA"].map(_is_specialized)
    on_all["_mandatory"] = (
        on_all["Programa de Valor"].map(_is_titan)
        | pd.to_numeric(on_all["Fenix"], errors="coerce").fillna(0).gt(0)
    )
    on_selected = on_all[on_all[config.SELECTION_COL].eq("T")].copy()
    for city, group in on_selected.groupby("NOMBRE"):
        total = len(group)
        specialized = int(group["_is_specialized"].sum())
        reference_90 = int(np.floor(total * config.ON_SPECIALIZED_TARGET + 0.5))
        rows.append({
            "Control": "Especialización ON (referencia secundaria)", "Canal": "ON", "NOMBRE": city, "Subcanal2": "ON",
            "ESTRATEGICA": "", "LOC.COM": "", "Objetivo": reference_90, "Titulares": specialized,
            "Diferencia": specialized - reference_90, "Referencia_90": reference_90,
            "Actual_%": round(specialized / total * 100, 2) if total else 0,
            "Estado": "OK" if specialized == reference_90 else "INFORMATIVO",
            "Motivo": "El rango 10–30, el balance y la cuota exacta prevalecen; 90/10 queda como referencia secundaria",
        })

    for (city, subchannel), group in on_all.groupby(["NOMBRE", "Subcanal2"], dropna=False):
        total_target = int(group[config.SELECTION_COL].eq("T").sum())
        if total_target <= 0:
            continue
        route_targets, _target_si = _route_specialized_targets(group, total_target)
        group_route_key = group["RUT.COM"].astype(str)
        for route_key, route_target in sorted(route_targets.items()):
            route_mask = group_route_key.eq(route_key)
            available = int(route_mask.sum())
            available_si = int((route_mask & group["_is_specialized"]).sum())
            actual = int((route_mask & group["_is_specialized"] & group[config.SELECTION_COL].eq("T")).sum())
            total_titles_route = int((route_mask & group[config.SELECTION_COL].eq("T")).sum())
            mandatory_route = int((route_mask & group["_is_specialized"] & group["_mandatory"]).sum())
            substitutes_route = int((route_mask & group[config.SELECTION_COL].isin(["S1", "S2", "S3"])).sum())
            if route_target == 0 and actual == 0:
                continue
            if actual != route_target:
                state = "INFORMATIVO" if actual > int(config.ON_SPECIALIZED_ROUTE_MAX) else "REVISAR"
                reason = (
                    "Excepción inevitable: no hay suficientes no especializados en rutas operativas para completar la cuota ON"
                    if actual > int(config.ON_SPECIALIZED_ROUTE_MAX)
                    else "La ruta no coincide con la distribución especializada balanceada calculada"
                )
            elif actual > int(config.ON_SPECIALIZED_ROUTE_MAX):
                state = "INFORMATIVO"
                reason = (
                    f"Excepción inevitable sobre {int(config.ON_SPECIALIZED_ROUTE_MAX)} por falta de puntos no especializados"
                )
            elif actual < int(config.ON_SPECIALIZED_ROUTE_ACTIVE_MIN):
                state = "INFORMATIVO"
                reason = (
                    f"Ruta especializada obligatoria con {mandatory_route} Titán/Fénix y solo {available_si} puntos SI; "
                    f"no puede alcanzar el mínimo de {int(config.ON_SPECIALIZED_ROUTE_ACTIVE_MIN)}"
                    if mandatory_route > 0
                    else (
                        "Excepción inevitable para completar la cuota ON: no existe capacidad suficiente "
                        "en rutas no especializadas"
                    )
                )
            elif substitutes_route == 0 and available > total_titles_route:
                state = "REVISAR"
                reason = "La ruta tiene candidatos disponibles pero quedó sin suplentes"
            elif substitutes_route == 0:
                mandatory_all = int((route_mask & group["_mandatory"]).sum())
                if mandatory_all >= total_titles_route and available == total_titles_route:
                    state = "INFORMATIVO"
                    reason = "Todos los puntos disponibles son titulares obligatorios; no existe candidato para suplente"
                else:
                    state = "REVISAR"
                    reason = "La selección consumió todos los candidatos de la ruta y no dejó suplente"
            else:
                state = "OK"
                reason = "Ruta especializada nivelada entre 10 y 30 y con suplentes disponibles"
            rows.append({
                "Control": "Titulares especializados por ruta ON (10–30 y balance)", "Canal": "ON", "NOMBRE": city,
                "Subcanal2": subchannel, "RUT.COM": route_key, "ESTRATEGICA": "", "LOC.COM": "",
                "Objetivo": route_target, "Titulares": actual, "Diferencia": actual - route_target,
                "Disponibles_SI": available_si, "Obligatorios_SI": mandatory_route, "Suplentes": substitutes_route,
                "Referencia_30": min(int(config.ON_SPECIALIZED_ROUTE_MAX), available_si),
                "Estado": state, "Motivo": reason,
            })

        comparable_targets = [
            target for target in route_targets.values()
            if target >= int(config.ON_SPECIALIZED_ROUTE_ACTIVE_MIN)
        ]
        comparable_actual = [
            int((group_route_key.eq(route) & group["_is_specialized"] & group[config.SELECTION_COL].eq("T")).sum())
            for route, target in route_targets.items()
            if target >= int(config.ON_SPECIALIZED_ROUTE_ACTIVE_MIN)
        ]
        target_gap = max(comparable_targets) - min(comparable_targets) if comparable_targets else 0
        actual_gap = max(comparable_actual) - min(comparable_actual) if comparable_actual else 0
        rows.append({
            "Control": "Nivelación de rutas especializadas ON por Nombre", "Canal": "ON", "NOMBRE": city,
            "Subcanal2": subchannel, "RUT.COM": "VARIAS", "ESTRATEGICA": "", "LOC.COM": "",
            "Objetivo": target_gap, "Titulares": actual_gap, "Diferencia": actual_gap - target_gap,
            "Estado": "OK" if actual_gap == target_gap else "REVISAR",
            "Motivo": "Las rutas comparables quedaron en el reparto entero más nivelado posible",
        })

    mandatory = final["Programa de Valor"].map(_is_titan) | pd.to_numeric(final["Fenix"], errors="coerce").fillna(0).gt(0)
    mandatory_selected = int((mandatory & final[config.SELECTION_COL].eq("T")).sum())
    mandatory_total = int(mandatory.sum())
    rows.append({
        "Control": "Titán/Titán Plus/Fénix obligatorios", "Canal": "TODOS", "NOMBRE": "VARIOS", "Subcanal2": "",
        "ESTRATEGICA": "", "LOC.COM": "", "Objetivo": mandatory_total, "Titulares": mandatory_selected,
        "Diferencia": mandatory_selected - mandatory_total,
        "Estado": "OK" if mandatory_selected == mandatory_total else "REVISAR",
        "Motivo": "Todos los puntos obligatorios son titulares" if mandatory_selected == mandatory_total else "Hay puntos obligatorios sin titularidad",
    })

    substitute_counts = final[config.SELECTION_COL].value_counts()
    target_substitutes = int(round(np.mean([int(substitute_counts.get(level, 0)) for level in ("S1", "S2", "S3")])))
    for level in ("S1", "S2", "S3"):
        actual = int(substitute_counts.get(level, 0))
        rows.append({
            "Control": "Balance de suplentes", "Canal": "TODOS", "NOMBRE": "VARIOS", "Subcanal2": "",
            "ESTRATEGICA": level, "LOC.COM": "", "Objetivo": target_substitutes, "Titulares": actual,
            "Diferencia": actual - target_substitutes,
            "Estado": "OK" if abs(actual - target_substitutes) <= max(1, round(target_substitutes * 0.10)) else "VARIACION",
            "Motivo": "S1 limitado para mantener niveles S1/S2/S3 comparables",
        })
    s4_count = int(substitute_counts.get("S4", 0))
    rows.append({
        "Control": "Ausencia de suplentes S4", "Canal": "TODOS", "NOMBRE": "VARIOS", "Subcanal2": "",
        "ESTRATEGICA": "S4", "LOC.COM": "", "Objetivo": 0, "Titulares": s4_count,
        "Diferencia": s4_count, "Estado": "OK" if s4_count == 0 else "REVISAR",
        "Motivo": "Solo se generan S1, S2 y S3" if s4_count == 0 else "Se encontraron asignaciones S4",
    })

    title_rows = final.loc[
        final[config.SELECTION_COL].eq("T"), ["_CANAL_SELECCION", "NOMBRE", "RUT.COM"]
    ]
    title_routes = set(
        (str(canal), str(city), str(route))
        for canal, city, route in title_rows.itertuples(index=False, name=None)
    )
    substitutes = final[final[config.SELECTION_COL].isin(["S1", "S2", "S3"])].copy()
    invalid_substitutes = int(sum(
        (str(canal), str(city), str(route)) not in title_routes
        for canal, city, route in substitutes[["_CANAL_SELECCION", "NOMBRE", "RUT.COM"]].itertuples(index=False, name=None)
    ))
    rows.append({
        "Control": "Suplentes solo en rutas con titulares", "Canal": "TODOS", "NOMBRE": "VARIOS",
        "Subcanal2": "", "ESTRATEGICA": "S1-S3", "LOC.COM": "", "Objetivo": 0,
        "Titulares": invalid_substitutes, "Diferencia": invalid_substitutes,
        "Estado": "OK" if invalid_substitutes == 0 else "REVISAR",
        "Motivo": "Ningún suplente está en una ruta sin T" if invalid_substitutes == 0 else "Hay suplentes asignados a rutas sin titular",
    })

    sequence_violations = []
    sequence_frame = final.assign(
        _SPECIALIZATION_CONTROL=final["ESPECIALIZADA"].map(
            lambda value: "SI" if _is_specialized(value) else "NO"
        )
    )
    for key, group in sequence_frame.groupby(
        ["_CANAL_SELECCION", "NOMBRE", "RUT.COM", "_SPECIALIZATION_CONTROL"],
        dropna=False,
    ):
        present = set(group[config.SELECTION_COL].astype(str).str.upper())
        has_gap = (
            "S2" in present and "S1" not in present
            or "S3" in present and not {"S1", "S2"}.issubset(present)
        )
        if has_gap:
            sequence_violations.append(key)
    rows.append({
        "Control": "Secuencia S1-S3 por ruta y especialización", "Canal": "TODOS",
        "NOMBRE": "VARIOS", "Subcanal2": "", "ESTRATEGICA": "S1-S3", "LOC.COM": "",
        "Objetivo": 0, "Titulares": len(sequence_violations),
        "Diferencia": len(sequence_violations),
        "Estado": "OK" if not sequence_violations else "REVISAR",
        "Motivo": (
            "Todas las rutas empiezan en S1 y continúan sin saltos"
            if not sequence_violations
            else "Hay rutas o especializaciones con niveles de suplente omitidos"
        ),
    })

    on_no = sequence_frame.loc[
        sequence_frame["_CANAL_SELECCION"].eq("ON")
        & sequence_frame["_SPECIALIZATION_CONTROL"].eq("NO")
    ].copy()
    optional_small_routes = 0
    uncovered_with_candidates = 0
    uncovered_without_candidates = 0
    for _key, group in on_no.groupby(["NOMBRE", "RUT.COM"], dropna=False):
        title_mask = group[config.SELECTION_COL].eq("T")
        if not bool(title_mask.any()):
            continue
        mandatory_mask = (
            group["Programa de Valor"].map(_is_titan)
            | pd.to_numeric(group["Fenix"], errors="coerce").fillna(0).gt(0)
        )
        if bool((title_mask & ~mandatory_mask).any()) and int(title_mask.sum()) < int(config.ON_NON_SPECIALIZED_ROUTE_MIN):
            optional_small_routes += 1
        has_substitute = bool(group[config.SELECTION_COL].isin(["S1", "S2", "S3"]).any())
        if not has_substitute:
            if bool((~title_mask).any()):
                uncovered_with_candidates += 1
            else:
                uncovered_without_candidates += 1
    rows.append({
        "Control": "Titulares NO agrupados por ruta ON", "Canal": "ON",
        "NOMBRE": "VARIOS", "Subcanal2": "ON", "ESTRATEGICA": "", "LOC.COM": "",
        "Objetivo": 0, "Titulares": optional_small_routes,
        "Diferencia": optional_small_routes,
        "Estado": "OK" if optional_small_routes == 0 else "INFORMATIVO",
        "Motivo": (
            f"Las rutas NO opcionales tienen al menos {config.ON_NON_SPECIALIZED_ROUTE_MIN} titulares; "
            "las rutas menores contienen únicamente puntos obligatorios"
            if optional_small_routes == 0
            else (
                "Excepciones NO menores usadas únicamente cuando son necesarias para conservar la cuota "
                f"y evitar superar {config.ON_SPECIALIZED_ROUTE_MAX} especializados por ruta"
            )
        ),
    })
    rows.append({
        "Control": "Suplentes para titulares NO por ruta ON", "Canal": "ON",
        "NOMBRE": "VARIOS", "Subcanal2": "ON", "ESTRATEGICA": "NO", "LOC.COM": "",
        "Objetivo": 0, "Titulares": uncovered_with_candidates,
        "Diferencia": uncovered_with_candidates, "Sin_candidato": uncovered_without_candidates,
        "Estado": "OK" if uncovered_with_candidates == 0 else "REVISAR",
        "Motivo": (
            f"Toda ruta con candidato NO disponible recibió suplente; {uncovered_without_candidates} "
            "ruta(s) no tienen ningún candidato adicional en el universo elegible"
            if uncovered_with_candidates == 0
            else "Hay rutas con titulares NO y candidatos disponibles que quedaron sin suplente"
        ),
    })

    on_title_routes = set(
        final.loc[
            final["_CANAL_SELECCION"].eq("ON") & final[config.SELECTION_COL].eq("T"), "RUT.COM"
        ].astype(str)
    )
    off_title_routes = set(
        final.loc[
            final["_CANAL_SELECCION"].eq("OFF") & final[config.SELECTION_COL].eq("T"), "RUT.COM"
        ].astype(str)
    )
    off_all = final.loc[final["_CANAL_SELECCION"].eq("OFF")]
    off_qualified_routes = set(
        off_all.loc[
            off_all["Programa de Valor"].map(_is_titan)
            | pd.to_numeric(off_all["Fenix"], errors="coerce").fillna(0).gt(0),
            "RUT.COM",
        ].astype(str)
    )
    compatible_on_routes = on_title_routes & off_qualified_routes
    covered_on_routes = len(compatible_on_routes & off_title_routes)
    uncovered_compatible = compatible_on_routes - off_title_routes
    without_compatible_off = on_title_routes - off_qualified_routes
    rows.append({
        "Control": "Rutas con titulares ON también cubiertas en OFF", "Canal": "TODOS", "NOMBRE": "VARIOS",
        "Subcanal2": "", "ESTRATEGICA": "", "LOC.COM": "", "Objetivo": len(compatible_on_routes),
        "Titulares": covered_on_routes, "Diferencia": covered_on_routes - len(compatible_on_routes),
        "Rutas_ON_totales": len(on_title_routes), "Sin_candidato_OFF": len(without_compatible_off),
        "Estado": "OK" if not uncovered_compatible else "INFORMATIVO",
        "Motivo": (
            f"Se cubrieron las {covered_on_routes} rutas ON compatibles en OFF; "
            f"{len(without_compatible_off)} no tienen el mismo RUT.COM dentro de una ruta Titán/Fénix OFF"
            if not uncovered_compatible
            else f"Faltan {len(uncovered_compatible)} rutas compatibles por cubrir sin romper las demás reglas"
        ),
    })

    off = final[final["_CANAL_SELECCION"].eq("OFF")]
    qualified_routes = set(
        off.loc[
            off["Programa de Valor"].map(_is_titan)
            | pd.to_numeric(off["Fenix"], errors="coerce").fillna(0).gt(0),
            "RUT.COM",
        ].astype(str)
    )
    selected_or_substitute = off[config.SELECTION_COL].isin(["T", "S1", "S2", "S3"])
    invalid_off_selections = int((selected_or_substitute & ~off["RUT.COM"].astype(str).isin(qualified_routes)).sum())
    rows.append({
        "Control": "Selecciones OFF solo en rutas Titán/Fénix", "Canal": "OFF", "NOMBRE": "VARIOS", "Subcanal2": "",
        "ESTRATEGICA": "T/S1/S2/S3", "LOC.COM": "", "Objetivo": 0, "Titulares": invalid_off_selections,
        "Diferencia": invalid_off_selections, "Estado": "OK" if invalid_off_selections == 0 else "REVISAR",
        "Motivo": "Todas las selecciones OFF pertenecen a rutas calificadas" if invalid_off_selections == 0 else "Hay titulares o suplentes OFF en rutas sin Titán/Fénix",
    })
    return pd.DataFrame(rows)


def run_selection_process(
    df_elegible: pd.DataFrame,
    historical: pd.DataFrame,
    total_target: int | None = None,
    increase_scope: str = "AMBOS",
):
    print("Running selection rules on the ELEGIBLE universe...")
    historical_city_targets, strategy_targets = _historical_targets(historical)
    historical_city_targets = _with_fixed_on_quotas(historical_city_targets)
    lima_quotas = config.load_lima_quotas()
    requested_total = int(total_target) if total_target is not None else int(historical_city_targets.sum())
    increase_scope = str(increase_scope).strip().upper()
    city_targets = _scaled_city_targets(
        df_elegible, historical_city_targets, requested_total, lima_quotas, increase_scope,
    )
    on_mask = df_elegible["Subcanal2"].eq("ON") | df_elegible["CANAL"].eq("ON")
    selected_on = select_canal_on(df_elegible.loc[on_mask], city_targets, strategy_targets)
    on_title_routes = set(
        selected_on.loc[selected_on[config.SELECTION_COL].eq("T"), "RUT.COM"].astype(str)
    )
    selected_off = select_canal_off(
        df_elegible.loc[~on_mask], city_targets, strategy_targets, lima_quotas,
        preferred_on_routes=on_title_routes,
    )
    selected_on["_CANAL_SELECCION"] = "ON"
    selected_off["_CANAL_SELECCION"] = "OFF"

    final = pd.concat([selected_off, selected_on], ignore_index=True)
    final["SEL_OFF"] = (final["_CANAL_SELECCION"].eq("OFF") & final[config.SELECTION_COL].eq("T")).astype(int)
    final["SEL_ON"] = (final["_CANAL_SELECCION"].eq("ON") & final[config.SELECTION_COL].eq("T")).astype(int)
    final["SEL_TOTAL"] = final["SEL_OFF"] + final["SEL_ON"]
    final["VISITAS"] = final["VISITAS_ANT"] + final["SEL_TOTAL"]
    actual_total = int(final[config.SELECTION_COL].eq("T").sum())
    actual_on = int((
        final["_CANAL_SELECCION"].eq("ON")
        & final[config.SELECTION_COL].eq("T")
    ).sum())
    if actual_total != requested_total:
        raise RuntimeError(
            f"La muestra debe sumar {requested_total:,} titulares, pero quedó en {actual_total:,}."
        )
    if actual_on != int(config.ON_FIXED_TOTAL):
        raise RuntimeError(
            f"Las cuotas ON deben sumar {int(config.ON_FIXED_TOTAL):,}, pero quedaron en {actual_on:,}."
        )
    controls = build_controls(
        final, city_targets, historical_city_targets, strategy_targets, lima_quotas,
        requested_total, increase_scope,
    )
    print(f"Selection complete. Total Titulares: {int(final[config.SELECTION_COL].eq('T').sum())}")
    print(controls["Estado"].value_counts().to_dict())
    return final, controls
