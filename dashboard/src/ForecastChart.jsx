/**
 * EnergyLens — ForecastChart component.
 *
 * Displays actual spot prices alongside ML ensemble forecasts
 * in a dual-line Recharts chart. Includes:
 *   - Actual prices (solid line)
 *   - Forecast prices (dashed line, distinct color)
 *   - Confidence indicator
 *   - Per-model breakdown on hover
 *   - Loading / error / "no models" states
 *
 * Integrates with the existing dashboard layout via props or
 * standalone fetch from /api/forecast + /api/prices.
 */

import { useCallback, useEffect, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import SignalQualityGate from './SignalQualityGate';

const API_BASE = import.meta.env.VITE_API_URL || '';

// ── Palette ─────────────────────────────────────────────────────────
const COLORS = {
  actual: "#2563eb",        // blue-600
  forecast: "#f59e0b",      // amber-500
  confidence: "#fef3c7",    // amber-50
  grid: "#e5e7eb",
  text: "#374151",
  textMuted: "#9ca3af",
  bg: "#ffffff",
  cardBg: "#f9fafb",
  border: "#e5e7eb",
  success: "#10b981",
  warning: "#f59e0b",
  error: "#ef4444",
};

// ── Helpers ─────────────────────────────────────────────────────────

function formatHour(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}

function formatDate(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function formatPrice(v) {
  if (v == null || isNaN(v)) return "—";
  return `€${v.toFixed(2)}`;
}

function confidenceLabel(c) {
  if (c >= 75) return { text: "High", color: COLORS.success };
  if (c >= 55) return { text: "Medium", color: COLORS.warning };
  return { text: "Low", color: COLORS.error };
}

// ── Custom Tooltip ──────────────────────────────────────────────────

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  const actual = payload.find((p) => p.dataKey === "actual");
  const forecast = payload.find((p) => p.dataKey === "forecast");

  return (
    <div
      style={{
        background: COLORS.bg,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 8,
        padding: "10px 14px",
        fontSize: 13,
        boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
        minWidth: 160,
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 6, color: COLORS.text }}>
        {label}
      </div>
      {actual && (
        <div style={{ display: "flex", justifyContent: "space-between", gap: 20, marginBottom: 3 }}>
          <span style={{ color: COLORS.actual }}>● Actual</span>
          <span style={{ fontWeight: 600 }}>{formatPrice(actual.value)}</span>
        </div>
      )}
      {forecast && (
        <div style={{ display: "flex", justifyContent: "space-between", gap: 20 }}>
          <span style={{ color: COLORS.forecast }}>◆ Forecast</span>
          <span style={{ fontWeight: 600 }}>{formatPrice(forecast.value)}</span>
        </div>
      )}
    </div>
  );
}

// ── Model breakdown panel ───────────────────────────────────────────

function ModelBreakdown({ perModel, modelsUsed, modelsTotal }) {
  if (!perModel || Object.keys(perModel).length === 0) return null;

  const sorted = Object.entries(perModel).sort((a, b) => b[1] - a[1]);
  const mean = sorted.reduce((s, [, v]) => s + v, 0) / sorted.length;

  return (
    <div
      style={{
        marginTop: 16,
        padding: "14px 18px",
        background: COLORS.cardBg,
        borderRadius: 8,
        border: `1px solid ${COLORS.border}`,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: COLORS.text,
          marginBottom: 10,
        }}
      >
        Per-model predictions ({modelsUsed}/{modelsTotal} models)
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "6px 24px",
        }}
      >
        {sorted.map(([name, price]) => {
          const diff = price - mean;
          const diffColor = Math.abs(diff) < 5 ? COLORS.textMuted : diff > 0 ? COLORS.error : COLORS.success;
          return (
            <div
              key={name}
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 12,
                padding: "3px 0",
                borderBottom: `1px solid ${COLORS.border}`,
              }}
            >
              <span style={{ color: COLORS.textMuted }}>{name.replace(/_/g, " ")}</span>
              <span>
                <span style={{ fontWeight: 600, color: COLORS.text }}>{formatPrice(price)}</span>
                <span style={{ color: diffColor, marginLeft: 6, fontSize: 11 }}>
                  {diff >= 0 ? "+" : ""}{diff.toFixed(1)}
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═════════════════════════════════════════════════════════════════════

export default function ForecastChart({ zone = "DK1", hours = 24, actualDays = 2 }) {
  const [chartData, setChartData] = useState([]);
  const [forecastMeta, setForecastMeta] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch actuals and forecast in parallel (catch network/CORS errors)
      const [pricesRes, forecastRes] = await Promise.all([
        fetch(`${API_BASE}/api/prices?zone=${zone}&days=${actualDays}`).catch(() => ({ ok: false })),
        fetch(`${API_BASE}/api/forecast?zone=${zone}&hours=${hours}`).catch(() => ({ ok: false })),
      ]);

      let actuals = [];
      if (pricesRes.ok) {
        const pricesData = await pricesRes.json();
        actuals = (pricesData.records || []).map((r) => ({
          timestamp: r.HourUTC || r.timestamp,
          label: formatHour(r.HourUTC || r.timestamp),
          actual: r.SpotPriceEUR ?? r.price_eur,
          forecast: null,
        }));
      }

      // Handle forecast (may be 503 if no models trained yet)
      let forecasts = [];
      let meta = null;

      if (forecastRes.ok) {
        const fData = await forecastRes.json();
        meta = {
          confidence: fData.confidence,
          modelsUsed: fData.models_used,
          modelsTotal: fData.models_total,
          perModel: fData.per_model,
          currentPrice: fData.current_price,
          priceRange: fData.price_range,
          qualityGate: fData.quality_gate,
          generatedAt: fData.generated_at,
        };

        forecasts = (fData.forecasts || []).map((f) => ({
          timestamp: f.timestamp_utc,
          label: formatHour(f.timestamp_utc),
          actual: null,
          forecast: f.price_eur,
        }));
      } else if (forecastRes.status === 503) {
        // No models yet — show actuals only with a message
        const body = await forecastRes.json().catch(() => ({}));
        meta = { noModels: true, message: body.detail || "No trained models available" };
      } else {
        throw new Error(`Forecast API: ${forecastRes.status}`);
      }

      // Overlap: last actual point gets the first forecast value too
      if (actuals.length > 0 && forecasts.length > 0) {
        const lastActual = actuals[actuals.length - 1];
        lastActual.forecast = lastActual.actual; // bridge point
      }

      setChartData([...actuals, ...forecasts]);
      setForecastMeta(meta);
    } catch (err) {
      console.error("ForecastChart fetch error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [zone, hours, actualDays]);

  useEffect(() => {
    fetchData();
    // Auto-refresh every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // ── Render states ───────────────────────────────────────────────

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: COLORS.textMuted }}>
        Loading forecast…
      </div>
    );
  }

  if (error) {
    return (
      <div
        style={{
          padding: 24,
          background: "#fef2f2",
          borderRadius: 8,
          border: "1px solid #fecaca",
          color: COLORS.error,
        }}
      >
        <strong>Forecast unavailable:</strong> {error}
        <div style={{ marginTop: 8 }}>
          <button
            onClick={fetchData}
            style={{
              padding: "6px 16px",
              borderRadius: 6,
              border: `1px solid ${COLORS.error}`,
              background: "transparent",
              color: COLORS.error,
              cursor: "pointer",
              fontSize: 13,
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const conf = forecastMeta?.confidence ? confidenceLabel(forecastMeta.confidence) : null;
  const noModels = forecastMeta?.noModels;

  return (
    <div>
      {/* Header row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: COLORS.text }}>
            {zone} Price Forecast
          </h3>
          <p style={{ margin: "4px 0 0", fontSize: 13, color: COLORS.textMuted }}>
            {actualDays}-day actuals + {hours}h forecast
          </p>
        </div>

        {conf && !noModels && (
          <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 13 }}>
            <div>
              <span style={{ color: COLORS.textMuted }}>Confidence </span>
              <span style={{ fontWeight: 700, color: conf.color }}>
                {forecastMeta.confidence}% ({conf.text})
              </span>
            </div>
            <div style={{ color: COLORS.textMuted }}>
              {forecastMeta.modelsUsed}/{forecastMeta.modelsTotal} models
            </div>
            {forecastMeta.currentPrice != null && (
              <div>
                <span style={{ color: COLORS.textMuted }}>Current </span>
                <span style={{ fontWeight: 600 }}>{formatPrice(forecastMeta.currentPrice)}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* No-models banner */}
      {noModels && (
        <div
          style={{
            padding: "12px 16px",
            marginBottom: 16,
            background: "#fffbeb",
            border: "1px solid #fde68a",
            borderRadius: 8,
            fontSize: 13,
            color: "#92400e",
          }}
        >
          {forecastMeta.message}. Run <code>python -m ml.run_training</code> to train models,
          then restart the API.
        </div>
      )}

      {/* Chart */}
      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={chartData} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.grid} />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: COLORS.textMuted }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 11, fill: COLORS.textMuted }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `€${v}`}
            domain={["auto", "auto"]}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            iconType="plainline"
          />

          {/* Actual prices */}
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual price"
            stroke={COLORS.actual}
            strokeWidth={2}
            dot={false}
            connectNulls={false}
          />

          {/* Forecast prices */}
          <Line
            type="monotone"
            dataKey="forecast"
            name="Forecast"
            stroke={COLORS.forecast}
            strokeWidth={2}
            strokeDasharray="6 3"
            dot={false}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Per-model breakdown */}
      {forecastMeta && !noModels && (
        <ModelBreakdown
          perModel={forecastMeta.perModel}
          modelsUsed={forecastMeta.modelsUsed}
          modelsTotal={forecastMeta.modelsTotal}
        />
      )}

      {/* Signal Quality Gate */}
        {forecastMeta?.qualityGate && (
          <SignalQualityGate
            qualityGate={forecastMeta.qualityGate}
            confidence={forecastMeta.confidence}
          />
        )}
    </div>
  );
}
