#!/usr/bin/env python3
"""
Apply ENTSO-E generation feature changes to EnergyLens.
Run from the EnergyLens root:  python install_generation_features.py
"""
import os, sys

def read(p):
    with open(p) as f: return f.read()
def write(p, c):
    with open(p, "w") as f: f.write(c)

n = 0

# ── 1. database.py: add get_generation_data() ────────────────────────
print("1. core/database.py ...")
db = read("core/database.py")
if "get_generation_data" not in db:
    method = (
        "\n"
        "    def get_generation_data(self, zone, start=None, end=None):\n"
        '        """Get generation data for a zone, optionally within a time range."""\n'
        "        with self.connection() as conn:\n"
        "            cur = conn.cursor()\n"
        '            ph = "%s" if self.use_postgres else "?"\n'
        "            sql = f\"SELECT valid_time, generation_type, value_mw, is_forecast FROM generation WHERE zone = {ph}\"\n"
        "            params = [zone]\n"
        "            if start:\n"
        '                sql += f" AND valid_time >= {ph}"\n'
        "                params.append(start)\n"
        "            if end:\n"
        '                sql += f" AND valid_time <= {ph}"\n'
        "                params.append(end)\n"
        '            sql += " ORDER BY valid_time"\n'
        "            cur.execute(sql, params)\n"
        "            columns = [d[0] for d in cur.description]\n"
        "            return [dict(zip(columns, row)) for row in cur.fetchall()]\n"
        "\n"
    )
    marker = "    def quarantine_record"
    i = db.find(marker)
    if i == -1:
        print("  WARN: quarantine_record not found")
    else:
        db = db[:i] + method + db[i:]
        write("core/database.py", db)
        n += 1
        print("  + get_generation_data()")
else:
    print("  already present")

# ── 2. pipeline/ingest.py: add _backfill_generation ─────────────────
print("2. pipeline/ingest.py ...")
ing = read("pipeline/ingest.py")
if "_backfill_generation" not in ing:
    method = (
        "    async def _backfill_generation(self, days: int) -> int:\n"
        '        """Backfill historical generation data from ENTSO-E."""\n'
        '        logger.info(f"Backfilling {days} days of generation data...")\n'
        "        from config.constants import ACTIVE_ZONES\n"
        "        from datetime import timedelta\n"
        "        total = 0\n"
        "        now = datetime.now(timezone.utc)\n"
        "        for zone in ACTIVE_ZONES:\n"
        "            for day_offset in range(days):\n"
        "                target = now - timedelta(days=day_offset)\n"
        "                try:\n"
        "                    records = await self.entsoe.get_generation_forecast(zone=zone, date=target)\n"
        "                    if records:\n"
        "                        total += self.db.insert_generation(records, zone)\n"
        "                    actual = await self.entsoe.get_actual_generation(zone=zone, date=target)\n"
        "                    if actual:\n"
        '                        for r in actual: r["is_forecast"] = False\n'
        "                        total += self.db.insert_generation(actual, zone)\n"
        "                    if day_offset % 30 == 0:\n"
        '                        logger.info(f"  Gen backfill: {zone} day {day_offset}/{days}, total={total}")\n'
        "                except Exception as e:\n"
        '                    logger.warning(f"Gen backfill failed {zone} day {day_offset}: {e}")\n'
        "                await asyncio.sleep(0.3)\n"
        '        logger.info(f"Generation backfill complete: {total} records")\n'
        "        return total\n"
        "\n"
    )
    # Insert before Health & Summary section
    marker = "    # ─── Health & Summary"
    if marker not in ing:
        marker = "    def get_health"
    i = ing.find(marker)
    if i == -1:
        print("  WARN: insertion point not found")
    else:
        ing = ing[:i] + method + ing[i:]
        # Also add to run_backfill
        bm = "await self._backfill_weather(days)\n"
        j = ing.find(bm)
        if j != -1:
            end = j + len(bm)
            ing = ing[:end] + "\n        # Historical generation (ENTSO-E)\n        await self._backfill_generation(days)\n" + ing[end:]
        write("pipeline/ingest.py", ing)
        n += 1
        print("  + _backfill_generation() + updated run_backfill()")
else:
    print("  already present")

# ── 3. ml/features.py: add generation features ──────────────────────
print("3. ml/features.py ...")
feat = read("ml/features.py")
if "wind_generation_mw" not in feat:
    gen_block = (
        "\n"
        "    # ── 7b. ENTSO-E generation features ─────────────────────────────\n"
        '    if "wind_generation_mw" in feat.columns:\n'
        '        wind_gen = feat["wind_generation_mw"]\n'
        '        feat["wind_gen_lag_1"] = wind_gen.shift(1)\n'
        '        feat["wind_gen_lag_24"] = wind_gen.shift(24)\n'
        '        feat["wind_gen_change_1h"] = wind_gen.diff(1)\n'
        '        feat["wind_gen_change_24h"] = wind_gen.diff(24)\n'
        '        feat["wind_gen_mean_6"] = wind_gen.rolling(6, min_periods=1).mean()\n'
        '        feat["wind_gen_mean_24"] = wind_gen.rolling(24, min_periods=1).mean()\n'
        '        feat["wind_gen_std_24"] = wind_gen.rolling(24, min_periods=1).std()\n'
        '        feat["price_wind_gen_ratio"] = prices / (wind_gen + 1e-8)\n'
        "\n"
        '    if "solar_generation_mw" in feat.columns:\n'
        '        solar_gen = feat["solar_generation_mw"]\n'
        '        feat["solar_gen_lag_1"] = solar_gen.shift(1)\n'
        '        feat["solar_gen_lag_24"] = solar_gen.shift(24)\n'
        '        feat["solar_gen_mean_6"] = solar_gen.rolling(6, min_periods=1).mean()\n'
        "\n"
        '    if "total_generation_mw" in feat.columns:\n'
        '        total_gen = feat["total_generation_mw"]\n'
        '        feat["total_gen_lag_1"] = total_gen.shift(1)\n'
        '        feat["total_gen_lag_24"] = total_gen.shift(24)\n'
        '        feat["total_gen_mean_24"] = total_gen.rolling(24, min_periods=1).mean()\n'
        '        feat["total_gen_change_24h"] = total_gen.diff(24)\n'
        '        feat["price_per_gen_mw"] = prices / (total_gen + 1e-8)\n'
        "\n"
    )
    # Insert before spike_indicator
    marker = "    feat['spike_indicator']"
    i = feat.find(marker)
    if i == -1:
        marker = "spike_indicator"
        i = feat.find(marker)
    if i == -1:
        print("  WARN: spike_indicator not found")
    else:
        # Find start of line
        line_start = feat.rfind("\n", 0, i) + 1
        feat = feat[:line_start] + gen_block + feat[line_start:]
        write("ml/features.py", feat)
        n += 1
        print("  + 22 generation features (wind/solar/total)")
else:
    print("  already present")

# ── 4. ml/run_training.py: replace load_data_from_sqlite ────────────
print("4. ml/run_training.py ...")
train = read("ml/run_training.py")
if "generation_type" not in train:
    # Find function boundaries
    func_start = train.find("def load_data_from_sqlite(")
    func_end = train.find("\ndef main():")
    if func_start == -1 or func_end == -1:
        print("  WARN: function boundaries not found")
    else:
        new_func = '''def load_data_from_sqlite(db_path: str, zone: str = "DK1") -> pd.DataFrame | None:
    """
    Load spot prices, weather, and generation data from SQLite database
    and merge into a single DataFrame indexed by hour.
    """
    db = Path(db_path)
    if not db.exists():
        logger.error(f"Database not found: {db}")
        return None

    conn = sqlite3.connect(str(db))

    # ── Load spot prices ─────────────────────────────────────────────
    try:
        # Try Phase 1 schema first (HourUTC, PriceArea)
        prices_df = pd.read_sql_query(
            """
            SELECT HourUTC, SpotPriceEUR, SpotPriceDKK
            FROM spot_prices
            WHERE PriceArea = ?
            ORDER BY HourUTC
            """,
            conn,
            params=(zone,),
            parse_dates=["HourUTC"],
        )
        prices_df = prices_df.set_index("HourUTC")
    except Exception:
        # Fall back to production schema (valid_time, zone)
        try:
            prices_df = pd.read_sql_query(
                """
                SELECT valid_time AS HourUTC,
                       price_eur_mwh AS SpotPriceEUR,
                       price_dkk_mwh AS SpotPriceDKK
                FROM spot_prices
                WHERE zone = ?
                ORDER BY valid_time
                """,
                conn,
                params=(zone,),
                parse_dates=["HourUTC"],
            )
            prices_df = prices_df.set_index("HourUTC")
            prices_df = prices_df.resample("h").mean().dropna()
        except Exception as e:
            logger.error(f"Failed to load prices: {e}")
            conn.close()
            return None

    logger.info(f"Loaded {len(prices_df)} price records for {zone}")

    # ── Load weather data ────────────────────────────────────────────
    weather_df = pd.DataFrame()
    for table, time_col in [("weather_data", "time"), ("weather_forecasts", "valid_time")]:
        try:
            cols_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_names = [r[1] for r in cols_info]
            select_parts = [f"{time_col} AS time"]
            for c in ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]:
                if c in col_names:
                    select_parts.append(c)
            if len(select_parts) > 1:
                q = "SELECT " + ", ".join(select_parts) + f" FROM {table} ORDER BY {time_col}"
                weather_df = pd.read_sql_query(q, conn, parse_dates=["time"])
                weather_df = weather_df.set_index("time")
                logger.info(f"Loaded {len(weather_df)} weather records from {table}")
                break
        except Exception:
            continue

    # ── Load generation data (ENTSO-E) ───────────────────────────────
    gen_df = pd.DataFrame()
    try:
        gen_raw = pd.read_sql_query(
            """
            SELECT valid_time AS time, generation_type, value_mw
            FROM generation
            WHERE zone = ?
            ORDER BY valid_time
            """,
            conn,
            params=(zone,),
            parse_dates=["time"],
        )
        if len(gen_raw) > 0:
            gen_pivot = gen_raw.pivot_table(
                index="time", columns="generation_type",
                values="value_mw", aggfunc="mean"
            )
            wind_cols = [c for c in gen_pivot.columns if "wind" in c.lower()]
            solar_cols = [c for c in gen_pivot.columns if "solar" in c.lower()]
            gen_df = pd.DataFrame(index=gen_pivot.index)
            if wind_cols:
                gen_df["wind_generation_mw"] = gen_pivot[wind_cols].sum(axis=1)
            if solar_cols:
                gen_df["solar_generation_mw"] = gen_pivot[solar_cols].sum(axis=1)
            renewable_cols = wind_cols + solar_cols
            if renewable_cols:
                gen_df["renewable_generation"] = gen_pivot[renewable_cols].sum(axis=1)
            gen_df["total_generation_mw"] = gen_pivot.sum(axis=1)
            for col in gen_pivot.columns:
                gen_df[f"gen_{col}"] = gen_pivot[col]
            gen_df = gen_df.resample("h").mean()
            logger.info(f"Loaded {len(gen_df)} generation records for {zone}")
        else:
            logger.warning(f"No generation data found for {zone}")
    except Exception as e:
        logger.warning(f"Generation data not available: {e}")

    conn.close()

    # ── Merge on hourly index ────────────────────────────────────────
    merged = prices_df
    if not weather_df.empty:
        weather_hourly = weather_df.resample("h").mean()
        merged = merged.join(weather_hourly, how="left")
    if not gen_df.empty:
        merged = merged.join(gen_df, how="left")
        logger.info(f"Added generation features: {list(gen_df.columns)}")
    merged = merged.sort_index().dropna(subset=["SpotPriceEUR"])
    logger.info(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged

'''
        train = train[:func_start] + new_func + train[func_end+1:]
        write("ml/run_training.py", train)
        n += 1
        print("  + Updated load_data_from_sqlite with generation data")
else:
    print("  already present")

# ── 5. api/forecast_service.py: add generation loading ──────────────
print("5. api/forecast_service.py ...")
fc = read("api/forecast_service.py")
if "generation_type" not in fc:
    gen_loading = (
        "\n"
        "        # Load generation data (ENTSO-E)\n"
        "        gen_hourly = pd.DataFrame()\n"
        "        try:\n"
        "            gen_raw = pd.read_sql_query(\n"
        '                """\n'
        "                SELECT valid_time AS time, generation_type, value_mw\n"
        "                FROM generation\n"
        "                WHERE zone = ?\n"
        "                ORDER BY valid_time DESC\n"
        "                LIMIT ?\n"
        '                """,\n'
        "                conn,\n"
        "                params=(zone, lookback_hours * 10),\n"
        '                parse_dates=["time"],\n'
        "            )\n"
        "            if len(gen_raw) > 0:\n"
        '                gen_raw = gen_raw.sort_values("time")\n'
        "                gen_pivot = gen_raw.pivot_table(\n"
        '                    index="time", columns="generation_type",\n'
        '                    values="value_mw", aggfunc="mean"\n'
        "                )\n"
        '                wind_cols = [c for c in gen_pivot.columns if "wind" in c.lower()]\n'
        '                solar_cols = [c for c in gen_pivot.columns if "solar" in c.lower()]\n'
        "                gen_hourly = pd.DataFrame(index=gen_pivot.index)\n"
        "                if wind_cols:\n"
        '                    gen_hourly["wind_generation_mw"] = gen_pivot[wind_cols].sum(axis=1)\n'
        "                if solar_cols:\n"
        '                    gen_hourly["solar_generation_mw"] = gen_pivot[solar_cols].sum(axis=1)\n'
        "                renewable_cols = wind_cols + solar_cols\n"
        "                if renewable_cols:\n"
        '                    gen_hourly["renewable_generation"] = gen_pivot[renewable_cols].sum(axis=1)\n'
        '                gen_hourly["total_generation_mw"] = gen_pivot.sum(axis=1)\n'
        "                for col in gen_pivot.columns:\n"
        '                    gen_hourly[f"gen_{col}"] = gen_pivot[col]\n'
        '                gen_hourly = gen_hourly.resample("h").mean()\n'
        '                logger.info(f"Loaded {len(gen_hourly)} generation rows for {zone}")\n'
        "        except Exception as e:\n"
        '            logger.debug(f"Generation data not available: {e}")\n'
        "\n"
    )
    # Find the conn.close() in _load_recent_data (line 145)
    # Insert gen loading BEFORE conn.close()
    # We need to find the right conn.close() - the one at line 145
    # Look for "conn.close()" that is followed by weather merge logic
    old_close_merge = "        conn.close()\n\n        if not weather_hourly.empty:"
    if old_close_merge not in fc:
        # Try other patterns
        old_close_merge = "        conn.close()\n        if not weather_hourly.empty:"
    if old_close_merge not in fc:
        # Try finding second conn.close()
        first = fc.find("conn.close()")
        second = fc.find("conn.close()", first + 1)
        if second != -1:
            # Find the merge block after this conn.close
            merge_start = fc.find("if not weather_hourly", second)
            if merge_start != -1:
                # Find start of conn.close line
                line_start = fc.rfind("\n", 0, second) + 1
                old_close_merge = fc[line_start:fc.find("\n", merge_start) + 1]

    if old_close_merge in fc:
        new_block = gen_loading + "        conn.close()\n\n        merged = prices_df\n\n        if not weather_hourly.empty:\n            merged = merged.join(weather_hourly, how=\"left\")\n\n        if not gen_hourly.empty:\n            merged = merged.join(gen_hourly, how=\"left\")"
        # Also need to remove the old "else: merged = prices_df"
        old_merge_block = old_close_merge
        # Find everything up to and including the else clause
        merge_end = fc.find("merged = prices_df", fc.find(old_close_merge))
        if merge_end != -1:
            line_end = fc.find("\n", merge_end) + 1
            old_full = fc[fc.find(old_close_merge[:20], fc.find("conn.close()", fc.find("conn.close()") + 1) - 5):line_end]
        
        # Simpler approach: just insert before conn.close and fix the merge
        # Find the second occurrence of conn.close()
        first_close = fc.find("            conn.close()")
        if first_close == -1:
            first_close = fc.find("        conn.close()")
        second_close = fc.find("conn.close()", first_close + 20)
        
        if second_close != -1:
            close_line_start = fc.rfind("\n", 0, second_close) + 1
            close_line_end = fc.find("\n", second_close) + 1
            
            # Insert gen loading before conn.close
            fc = fc[:close_line_start] + gen_loading + fc[close_line_start:]
            
            # Now fix the merge section: find the weather merge after our insertion
            new_second_close = fc.find("conn.close()", fc.find("gen_hourly") + 10)
            merge_section_start = fc.find("if not weather_hourly", new_second_close)
            if merge_section_start != -1:
                # Find the line "merged = prices_df" after it
                prices_df_line = fc.find("merged = prices_df", merge_section_start)
                if prices_df_line != -1:
                    after_prices = fc.find("\n", prices_df_line) + 1
                    # Find where this merge block ends (next blank line + method or attribute)
                    # Insert gen_hourly join after weather join
                    join_line = fc.find('merged = merged.join(weather_hourly', merge_section_start)
                    if join_line == -1:
                        join_line = fc.find('merged = prices_df.join(weather_hourly', merge_section_start)
                    
                    if join_line != -1:
                        join_end = fc.find("\n", join_line) + 1
                        # Check if else clause follows
                        rest = fc[join_end:join_end+100].lstrip()
                        if rest.startswith("else:"):
                            else_end = fc.find("\n", fc.find("merged = prices_df", join_end)) + 1
                            # Replace the if/else with simpler version
                            ms_line_start = fc.rfind("\n", 0, merge_section_start) + 1
                            new_merge = (
                                "\n        merged = prices_df\n"
                                "\n        if not weather_hourly.empty:\n"
                                "            merged = merged.join(weather_hourly, how=\"left\")\n"
                                "\n        if not gen_hourly.empty:\n"
                                "            merged = merged.join(gen_hourly, how=\"left\")\n"
                            )
                            fc = fc[:ms_line_start] + new_merge + fc[else_end:]
            
            write("api/forecast_service.py", fc)
            n += 1
            print("  + Added generation data loading to forecast_service")
        else:
            print("  WARN: could not find second conn.close()")
    else:
        print("  WARN: could not find merge block pattern")
else:
    print("  already present")

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Applied {n} changes")
print(f"{'='*50}")
if n > 0:
    print("""
Next steps:
  1. python backfill_generation.py --days 365
  2. python backfill_generation.py --check
  3. python -m ml.run_training --zone DK1
  4. Upload models to GCS + redeploy
""")
