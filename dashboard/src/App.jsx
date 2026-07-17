import { useCallback, useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ForecastChart from './ForecastChart';
import AccuracyTracker from './AccuracyTracker';
import BacktestDashboard from './BacktestDashboard';
import ShapExplainer from './ShapExplainer';

// --- Design System ---
const COLORS = {
  bg: "#0a0e17",
  surface: "#111827",
  surfaceLight: "#1a2234",
  border: "#1e2a3a",
  borderLight: "#2a3a4e",
  text: "#e2e8f0",
  textMuted: "#8892a4",
  textDim: "#5a6478",
  dk1: "#22d3ee",       // Cyan — Denmark West
  dk1Dim: "rgba(34,211,238,0.12)",
  dk2: "#a78bfa",       // Purple — Denmark East
  dk2Dim: "rgba(167,139,250,0.12)",
  positive: "#34d399",
  warning: "#fbbf24",
  negative: "#f87171",
  accent: "#3b82f6",
};

const API_BASE = (import.meta.env.VITE_API_URL || '') + '/api';

// --- Components ---

function StatusDot({ status }) {
  const color = status === "LIVE" ? COLORS.positive
    : status === "STALE" ? COLORS.warning
    : COLORS.negative;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span style={{
        width: 8, height: 8, borderRadius: "50%", backgroundColor: color,
        boxShadow: `0 0 8px ${color}`,
        animation: status === "LIVE" ? "pulse 2s infinite" : "none",
      }} />
      <span style={{ fontSize: 11, fontWeight: 600, color, letterSpacing: "0.05em" }}>{status}</span>
    </span>
  );
}

function StaleBanner({ newest, onRefresh }) {
  if (!newest) return null;

  const newestDate = new Date(newest);
  const now = new Date();
  const ageHours = Math.round((now - newestDate) / (1000 * 60 * 60));
  const ageDays = Math.round(ageHours / 24);
  const ageText = ageDays > 1 ? `${ageDays} days old` : `${ageHours}h old`;

  return (
    <div style={{
      background: "rgba(251, 191, 36, 0.08)",
      border: `1px solid rgba(251, 191, 36, 0.25)`,
      borderRadius: 8,
      padding: "10px 16px",
      marginBottom: 14,
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 16 }}>&#9888;</span>
        <div>
          <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.warning }}>
            Showing historical data
          </span>
          <span style={{ fontSize: 12, color: COLORS.textMuted, marginLeft: 8 }}>
            Latest record: {newestDate.toLocaleDateString("da-DK")} ({ageText})
          </span>
        </div>
      </div>
      {onRefresh && (
        <button onClick={onRefresh} style={{
          padding: "4px 12px", borderRadius: 6, border: `1px solid ${COLORS.warning}`,
          background: "transparent", color: COLORS.warning, fontSize: 11,
          fontWeight: 600, cursor: "pointer",
        }}>
          Refresh
        </button>
      )}
    </div>
  );
}

function MetricCard({ label, value, sub, color, icon }) {
  return (
    <div style={{
      background: COLORS.surface, border: `1px solid ${COLORS.border}`,
      borderRadius: 10, padding: "18px 20px", flex: 1, minWidth: 180,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontSize: 11, color: COLORS.textMuted, fontWeight: 500, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: color || COLORS.text, letterSpacing: "-0.02em", lineHeight: 1.1 }}>{value}</div>
          {sub && <div style={{ fontSize: 12, color: COLORS.textMuted, marginTop: 4 }}>{sub}</div>}
        </div>
        {icon && <div style={{ fontSize: 24, opacity: 0.5 }}>{icon}</div>}
      </div>
    </div>
  );
}

function SectionHeader({ title, right }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, marginTop: 28 }}>
      <h2 style={{ fontSize: 15, fontWeight: 700, color: COLORS.text, margin: 0, letterSpacing: "-0.01em" }}>{title}</h2>
      {right}
    </div>
  );
}

function ZoneToggle({ active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 4, background: COLORS.surfaceLight, borderRadius: 8, padding: 3 }}>
      {["DK1", "DK2", "Both"].map(z => (
        <button key={z} onClick={() => onChange(z)} style={{
          padding: "5px 14px", borderRadius: 6, border: "none", cursor: "pointer",
          fontSize: 12, fontWeight: 600, transition: "all 0.2s",
          background: active === z ? (z === "DK1" ? COLORS.dk1Dim : z === "DK2" ? COLORS.dk2Dim : COLORS.accent + "22") : "transparent",
          color: active === z ? (z === "DK1" ? COLORS.dk1 : z === "DK2" ? COLORS.dk2 : COLORS.accent) : COLORS.textDim,
        }}>
          {z === "DK1" ? "DK1 West" : z === "DK2" ? "DK2 East" : "Compare"}
        </button>
      ))}
    </div>
  );
}

function TimeRangeSelector({ active, onChange }) {
  return (
    <div style={{ display: "flex", gap: 4, background: COLORS.surfaceLight, borderRadius: 8, padding: 3 }}>
      {[
        { key: "24h", label: "24H" },
        { key: "48h", label: "48H" },
        { key: "7d", label: "7D" },
        { key: "30d", label: "30D" },
      ].map(t => (
        <button key={t.key} onClick={() => onChange(t.key)} style={{
          padding: "5px 12px", borderRadius: 6, border: "none", cursor: "pointer",
          fontSize: 11, fontWeight: 600,
          background: active === t.key ? COLORS.accent + "22" : "transparent",
          color: active === t.key ? COLORS.accent : COLORS.textDim,
        }}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

function ConnectorRow({ name, status, records, lastUpdate }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "12px 16px", borderBottom: `1px solid ${COLORS.border}`,
    }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.text }}>{name}</div>
        <div style={{ fontSize: 11, color: COLORS.textDim, marginTop: 2 }}>{lastUpdate}</div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <span style={{ fontSize: 12, color: COLORS.textMuted }}>{records.toLocaleString()} records</span>
        <StatusDot status={status} />
      </div>
    </div>
  );
}

function QualityGateBar({ passed, failed, warnings }) {
  const total = passed + failed + warnings;
  if (total === 0) return null;
  return (
    <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", background: COLORS.surfaceLight }}>
      <div style={{ width: `${(passed/total)*100}%`, background: COLORS.positive, transition: "width 0.5s" }} />
      <div style={{ width: `${(warnings/total)*100}%`, background: COLORS.warning, transition: "width 0.5s" }} />
      <div style={{ width: `${(failed/total)*100}%`, background: COLORS.negative, transition: "width 0.5s" }} />
    </div>
  );
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: COLORS.surface + "f0", border: `1px solid ${COLORS.borderLight}`,
      borderRadius: 8, padding: "10px 14px", backdropFilter: "blur(12px)",
    }}>
      <div style={{ fontSize: 11, color: COLORS.textDim, marginBottom: 6 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: p.color }} />
          <span style={{ fontSize: 12, color: COLORS.textMuted }}>{p.name}:</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.text }}>
            {typeof p.value === "number" ? `\u20AC${p.value.toFixed(2)}` : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}

// --- Main Dashboard ---

export default function EnergyLensDashboard() {
  const [activeZone, setActiveZone] = useState("Both");
  const [activeTab, setActiveTab] = useState("forecast");
  const [timeRange, setTimeRange] = useState("48h");
  const [dk1Data, setDk1Data] = useState([]);
  const [dk2Data, setDk2Data] = useState([]);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [apiConnected, setApiConnected] = useState(false);
  const [dataStatus, setDataStatus] = useState("unknown"); // "fresh" | "stale" | "empty" | "offline"
  const [newestRecord, setNewestRecord] = useState(null);

  const loadData = useCallback(async () => {
    try {
      const days = timeRange === "24h" ? 1 : timeRange === "48h" ? 2 : timeRange === "7d" ? 7 : 30;

      const [priceRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/prices/compare?days=${days}`),
        fetch(`${API_BASE}/health`),
      ]);

      const prices = await priceRes.json();
      const h = await healthRes.json();

      // Extract data status from API response
      const meta = prices._meta || {};
      const status = meta.data_status || "fresh";
      setDataStatus(status);
      setApiConnected(true);

      // Get newest record timestamp from health endpoint
      if (h.data_range?.newest) {
        setNewestRecord(h.data_range.newest);
      }

      // Map price records (works for both fresh and stale fallback)
      const mapRecords = (records, zone) => (records || []).map(r => ({
        time: r.valid_time,
        label: new Date(r.valid_time).toLocaleString("da-DK", {
          day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit"
        }),
        price: r.price_eur_mwh,
        zone,
      }));

      setDk1Data(mapRecords(prices.DK1, "DK1"));
      setDk2Data(mapRecords(prices.DK2, "DK2"));
      setHealth(h.database);

    } catch (err) {
      console.error("API unreachable:", err);
      setApiConnected(false);
      setDataStatus("offline");
    }

    setLoading(false);
  }, [timeRange]);

  useEffect(() => { loadData(); }, [loadData]);

  // Auto-refresh every 30s
  useEffect(() => {
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  // Derive the display status for the header
  const displayStatus = !apiConnected ? "OFFLINE"
    : dataStatus === "fresh" ? "LIVE"
    : dataStatus === "stale" ? "STALE"
    : "OFFLINE";

  // Merge data for comparison chart
  const mergedData = dk1Data.map((d, i) => ({
    label: d.label,
    DK1: d.price,
    DK2: dk2Data[i]?.price ?? null,
  }));

  // Stats
  const dk1Prices = dk1Data.map(d => d.price).filter(Boolean);
  const dk2Prices = dk2Data.map(d => d.price).filter(Boolean);
  const currentDK1 = dk1Prices[dk1Prices.length - 1];
  const currentDK2 = dk2Prices[dk2Prices.length - 1];
  const avgDK1 = dk1Prices.length ? dk1Prices.reduce((a, b) => a + b, 0) / dk1Prices.length : 0;
  const spread = currentDK1 && currentDK2 ? Math.abs(currentDK1 - currentDK2) : 0;

  const qualityStats = health ? {
    passed: (health.spot_prices || 0) + (health.weather_forecasts || 0),
    failed: health.quality_quarantine || 0,
    warnings: Math.floor((health.spot_prices || 0) * 0.02),
  } : { passed: 0, failed: 0, warnings: 0 };

  if (loading) {
    return (
      <div style={{ background: COLORS.bg, minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ color: COLORS.textMuted, fontSize: 14 }}>Loading EnergyLens...</div>
      </div>
    );
  }

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", color: COLORS.text, fontFamily: "'Inter', -apple-system, sans-serif" }}>
      {/* --- Top Bar --- */}
      <header style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        padding: "14px 28px", borderBottom: `1px solid ${COLORS.border}`,
        background: COLORS.surface + "cc", backdropFilter: "blur(12px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: `linear-gradient(135deg, ${COLORS.dk1}, ${COLORS.dk2})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, fontWeight: 800, color: COLORS.bg,
          }}>E</div>
          <div>
            <span style={{ fontSize: 16, fontWeight: 800, letterSpacing: "-0.02em" }}>EnergyLens</span>
            <span style={{ fontSize: 10, color: COLORS.textDim, marginLeft: 8, fontWeight: 500, letterSpacing: "0.08em", textTransform: "uppercase" }}>Nordic Power Markets</span>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <StatusDot status={displayStatus} />
          <span style={{ fontSize: 11, color: COLORS.textDim }}>
            {new Date().toLocaleString("da-DK", { hour: "2-digit", minute: "2-digit", second: "2-digit" })} CET
          </span>
        </div>
      </header>

      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "20px 28px 60px" }}>

        {/* --- Stale Data Banner --- */}
        {dataStatus === "stale" && (
          <StaleBanner newest={newestRecord} onRefresh={loadData} />
        )}

        {/* --- Offline Banner --- */}
        {!apiConnected && (
          <div style={{
            background: "rgba(248, 113, 113, 0.08)",
            border: "1px solid rgba(248, 113, 113, 0.25)",
            borderRadius: 8, padding: "10px 16px", marginBottom: 14,
            display: "flex", alignItems: "center", gap: 10,
          }}>
            <span style={{ fontSize: 16 }}>&#10060;</span>
            <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.negative }}>
              API offline — start the server: uvicorn api.main:app --reload --port 8000
            </span>
          </div>
        )}

        {/* --- Tab Navigation --- */}
        <div style={{
          display: 'flex',
          gap: 4,
          marginBottom: 20,
          background: COLORS.surfaceLight,
          borderRadius: 8,
          padding: 3,
        }}>
          {[
            { id: 'forecast', label: '\u26A1 Forecast' },
            { id: 'accuracy', label: '\uD83C\uDFAF Accuracy' },
            { id: 'backtest', label: '\uD83D\uDCCA Backtest' },
            { id: 'explain', label: '\uD83D\uDD0D Explainability' },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '8px 16px',
                fontSize: 13,
                borderRadius: 6,
                border: 'none',
                background: activeTab === tab.id ? COLORS.accent + '22' : 'transparent',
                color: activeTab === tab.id ? COLORS.accent : COLORS.textDim,
                cursor: 'pointer',
                fontWeight: activeTab === tab.id ? 700 : 500,
                flex: 1,
                transition: 'all 0.2s',
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* ══════ FORECAST TAB ══════ */}
        {activeTab === 'forecast' && (
          <>
            {/* --- Metric Cards --- */}
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 8 }}>
              <MetricCard
                label="DK1 Current"
                value={currentDK1 ? `\u20AC${currentDK1.toFixed(2)}` : "\u2014"}
                sub={`EUR/MWh \u2014 West Denmark${dataStatus === "stale" ? " (historical)" : ""}`}
                color={COLORS.dk1}
                icon={"\u26A1"}
              />
              <MetricCard
                label="DK2 Current"
                value={currentDK2 ? `\u20AC${currentDK2.toFixed(2)}` : "\u2014"}
                sub={`EUR/MWh \u2014 East Denmark${dataStatus === "stale" ? " (historical)" : ""}`}
                color={COLORS.dk2}
                icon={"\u26A1"}
              />
              <MetricCard
                label="Zone Spread"
                value={`\u20AC${spread.toFixed(2)}`}
                sub="DK1\u2013DK2 differential"
                color={spread > 10 ? COLORS.warning : COLORS.textMuted}
                icon={"\u2194"}
              />
              <MetricCard
                label="24h Average"
                value={`\u20AC${avgDK1.toFixed(2)}`}
                sub={`${dk1Prices.length} data points`}
                icon={"\u03BC"}
              />
            </div>

            {/* --- Price Chart --- */}
            <SectionHeader
              title={`Spot Prices${dataStatus === "stale" ? " (Historical)" : ""}`}
              right={
                <div style={{ display: "flex", gap: 10 }}>
                  <ZoneToggle active={activeZone} onChange={setActiveZone} />
                  <TimeRangeSelector active={timeRange} onChange={setTimeRange} />
                </div>
              }
            />

            <div style={{
              background: COLORS.surface, border: `1px solid ${COLORS.border}`,
              borderRadius: 12, padding: "20px 16px 12px",
            }}>
              <ResponsiveContainer width="100%" height={340}>
                {activeZone === "Both" ? (
                  <LineChart data={mergedData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                    <XAxis dataKey="label" tick={{ fill: COLORS.textDim, fontSize: 10 }} tickLine={false} axisLine={{ stroke: COLORS.border }} interval={Math.max(1, Math.floor(mergedData.length / 12))} />
                    <YAxis tick={{ fill: COLORS.textDim, fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => `\u20AC${v}`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ fontSize: 12, color: COLORS.textMuted }} />
                    <Line type="monotone" dataKey="DK1" stroke={COLORS.dk1} strokeWidth={2} dot={false} name="DK1 West" />
                    <Line type="monotone" dataKey="DK2" stroke={COLORS.dk2} strokeWidth={2} dot={false} name="DK2 East" />
                  </LineChart>
                ) : (
                  <AreaChart data={activeZone === "DK1" ? dk1Data : dk2Data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={activeZone === "DK1" ? COLORS.dk1 : COLORS.dk2} stopOpacity={0.3} />
                        <stop offset="100%" stopColor={activeZone === "DK1" ? COLORS.dk1 : COLORS.dk2} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                    <XAxis dataKey="label" tick={{ fill: COLORS.textDim, fontSize: 10 }} tickLine={false} axisLine={{ stroke: COLORS.border }} interval={Math.max(1, Math.floor(dk1Data.length / 12))} />
                    <YAxis tick={{ fill: COLORS.textDim, fontSize: 10 }} tickLine={false} axisLine={false} tickFormatter={v => `\u20AC${v}`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Area type="monotone" dataKey="price" stroke={activeZone === "DK1" ? COLORS.dk1 : COLORS.dk2} strokeWidth={2} fill="url(#priceGrad)" name={`${activeZone} Price`} />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>

            {/* --- ML Forecast --- */}
            <SectionHeader title="Price Forecast" right={<StatusDot status={displayStatus} />} />
            <div style={{
              background: COLORS.surface, border: `1px solid ${COLORS.border}`,
              borderRadius: 12, padding: "20px 16px 12px",
            }}>
              <ForecastChart zone={activeZone === "Both" ? "DK1" : activeZone} hours={24} actualDays={2} />
            </div>

            {/* --- Pipeline Health + Quality Gate --- */}
            <div style={{ display: "flex", gap: 14, marginTop: 8 }}>
              {/* Connectors */}
              <div style={{ flex: 1 }}>
                <SectionHeader title="Data Pipeline" />
                <div style={{
                  background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                  borderRadius: 12, overflow: "hidden",
                }}>
                  <ConnectorRow
                    name="Nord Pool — Spot Prices"
                    status={dataStatus === "fresh" ? "LIVE" : dataStatus === "stale" ? "STALE" : "OFFLINE"}
                    records={health?.spot_prices || 0}
                    lastUpdate={dataStatus === "fresh" ? "Updated 30s ago" : newestRecord ? `Last data: ${new Date(newestRecord).toLocaleDateString("da-DK")}` : "No data"}
                  />
                  <ConnectorRow
                    name="Open-Meteo — Weather Forecasts"
                    status={health?.weather_forecasts > 0 ? "LIVE" : "STALE"}
                    records={health?.weather_forecasts || 0}
                    lastUpdate="Weather features"
                  />
                  <ConnectorRow
                    name="ENTSO-E — Generation Data"
                    status={health?.generation > 0 ? "LIVE" : "STALE"}
                    records={health?.generation || 0}
                    lastUpdate={health?.generation > 0 ? "Generation mix" : "Set ENTSOE_API_KEY in .env"}
                  />
                  <div style={{ padding: "14px 16px", borderTop: `1px solid ${COLORS.border}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.text }}>Quality Gate</span>
                      <span style={{ fontSize: 11, color: COLORS.textMuted }}>
                        {qualityStats.passed.toLocaleString()} passed &middot; {qualityStats.warnings} warnings &middot; {qualityStats.failed} quarantined
                      </span>
                    </div>
                    <QualityGateBar {...qualityStats} />
                  </div>
                </div>
              </div>

              {/* Database Stats */}
              <div style={{ width: 320 }}>
                <SectionHeader title="Database" />
                <div style={{
                  background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                  borderRadius: 12, padding: 16,
                }}>
                  {health && Object.entries(health).map(([table, count]) => (
                    <div key={table} style={{
                      display: "flex", justifyContent: "space-between", padding: "8px 0",
                      borderBottom: `1px solid ${COLORS.border}`,
                    }}>
                      <span style={{ fontSize: 12, color: COLORS.textMuted }}>{table.replace(/_/g, " ")}</span>
                      <span style={{ fontSize: 13, fontWeight: 700, color: count > 0 ? COLORS.text : COLORS.textDim, fontVariantNumeric: "tabular-nums" }}>
                        {count.toLocaleString()}
                      </span>
                    </div>
                  ))}

                  <div style={{ marginTop: 16, padding: "12px 14px", background: COLORS.surfaceLight, borderRadius: 8 }}>
                    <div style={{ fontSize: 11, color: COLORS.textDim, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 500 }}>Bitemporal Layer</div>
                    <div style={{ fontSize: 12, color: COLORS.textMuted, lineHeight: 1.5 }}>
                      Every record stored with <span style={{ color: COLORS.dk1, fontWeight: 600 }}>valid_time</span> and <span style={{ color: COLORS.dk2, fontWeight: 600 }}>knowledge_time</span>.
                      Point-in-time queries enabled for exact historical replay.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* ══════ ACCURACY TAB ══════ */}
        {activeTab === 'accuracy' && (
          <div style={{
            background: COLORS.surface, border: `1px solid ${COLORS.border}`,
            borderRadius: 12, padding: "20px 16px",
          }}>
            <AccuracyTracker zone={activeZone === "Both" ? "DK1" : activeZone} />
          </div>
        )}

        {/* ══════ BACKTEST TAB ══════ */}
        {activeTab === 'backtest' && (
          <div style={{
            background: COLORS.surface, border: `1px solid ${COLORS.border}`,
            borderRadius: 12, padding: "20px 16px",
          }}>
            <BacktestDashboard zone={activeZone === "Both" ? "DK1" : activeZone} />
          </div>
        )}

        {/* ══════ EXPLAINABILITY TAB ══════ */}
        {activeTab === 'explain' && (
          <div style={{
            background: COLORS.surface, border: `1px solid ${COLORS.border}`,
            borderRadius: 12, padding: "20px 16px",
          }}>
            <ShapExplainer zone={activeZone === "Both" ? "DK1" : activeZone} />
          </div>
        )}

        {/* --- Footer --- */}
        <div style={{ marginTop: 40, padding: "16px 0", borderTop: `1px solid ${COLORS.border}`, display: "flex", justifyContent: "space-between" }}>
          <span style={{ fontSize: 11, color: COLORS.textDim }}>EnergyLens v0.4.0 &mdash; ITU Consultant</span>
          <span style={{ fontSize: 11, color: COLORS.textDim }}>Data: Nord Pool &middot; ENTSO-E &middot; Open-Meteo</span>
        </div>
      </main>
    </div>
  );
}
