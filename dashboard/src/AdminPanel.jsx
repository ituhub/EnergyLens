/**
 * EnergyLens — Admin Panel Component.
 *
 * Renders inside the Admin tab (visible only to admin users).
 * Shows: user list, system stats, forecast logs, pipeline activity.
 *
 * Props:
 *   token — Firebase ID token for authenticated API calls
 */

import { useState, useEffect, useCallback } from "react";

const COLORS = {
  bg: "#0a0e17",
  surface: "#111827",
  surfaceLight: "#1a2234",
  border: "#1e2a3a",
  borderLight: "#2a3a4e",
  text: "#e2e8f0",
  textMuted: "#8892a4",
  textDim: "#5a6478",
  dk1: "#22d3ee",
  dk2: "#a78bfa",
  accent: "#3b82f6",
  positive: "#34d399",
  negative: "#f87171",
  warning: "#fbbf24",
};

const API_BASE = (import.meta.env.VITE_API_URL || "") + "/api";

export default function AdminPanel({ token }) {
  const [activeSection, setActiveSection] = useState("stats");
  const [stats, setStats] = useState(null);
  const [users, setUsers] = useState(null);
  const [logs, setLogs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const headers = { Authorization: `Bearer ${token}` };

  const fetchData = useCallback(async (section) => {
    setLoading(true);
    setError(null);
    try {
      if (section === "stats" || section === "all") {
        const res = await fetch(`${API_BASE}/admin/stats`, { headers });
        if (!res.ok) throw new Error(`Stats: ${res.status}`);
        setStats(await res.json());
      }
      if (section === "users" || section === "all") {
        const res = await fetch(`${API_BASE}/admin/users`, { headers });
        if (!res.ok) throw new Error(`Users: ${res.status}`);
        setUsers(await res.json());
      }
      if (section === "logs" || section === "all") {
        const res = await fetch(`${API_BASE}/admin/forecast-logs?hours=168&limit=100`, { headers });
        if (!res.ok) throw new Error(`Logs: ${res.status}`);
        setLogs(await res.json());
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    fetchData("all");
  }, [fetchData]);

  const toggleUser = async (uid, currentActive) => {
    try {
      await fetch(`${API_BASE}/admin/users/${uid}/toggle?active=${!currentActive}`, {
        method: "POST",
        headers,
      });
      fetchData("users");
    } catch (err) {
      setError(err.message);
    }
  };

  const SectionButton = ({ id, label, icon }) => (
    <button
      onClick={() => setActiveSection(id)}
      style={{
        padding: "8px 16px",
        fontSize: 12,
        fontWeight: activeSection === id ? 700 : 500,
        borderRadius: 6,
        border: "none",
        background: activeSection === id ? COLORS.accent + "22" : "transparent",
        color: activeSection === id ? COLORS.accent : COLORS.textDim,
        cursor: "pointer",
        transition: "all 0.2s",
      }}
    >
      {icon} {label}
    </button>
  );

  const StatCard = ({ label, value, sub, color }) => (
    <div
      style={{
        background: COLORS.surfaceLight,
        borderRadius: 10,
        padding: "16px 18px",
        flex: 1,
        minWidth: 140,
      }}
    >
      <div style={{ fontSize: 10, color: COLORS.textDim, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 24, fontWeight: 700, color: color || COLORS.text, fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: COLORS.textDim, marginTop: 4 }}>{sub}</div>}
    </div>
  );

  return (
    <div
      style={{
        background: COLORS.surface,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 12,
        padding: "20px 16px",
      }}
    >
      {/* Section nav */}
      <div style={{ display: "flex", gap: 4, marginBottom: 20, background: COLORS.bg, borderRadius: 8, padding: 3 }}>
        <SectionButton id="stats" label="System" icon="📊" />
        <SectionButton id="users" label="Users" icon="👥" />
        <SectionButton id="logs" label="Forecast Logs" icon="📋" />
      </div>

      {error && (
        <div style={{
          background: "rgba(248,113,113,0.08)",
          border: "1px solid rgba(248,113,113,0.25)",
          borderRadius: 8, padding: "10px 14px", marginBottom: 16,
          fontSize: 12, color: COLORS.negative,
        }}>
          {error}
        </div>
      )}

      {loading && !stats && (
        <div style={{ textAlign: "center", padding: 40, color: COLORS.textDim }}>Loading...</div>
      )}

      {/* ══════ SYSTEM STATS ══════ */}
      {activeSection === "stats" && stats && (
        <>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
            <StatCard label="Total Users" value={stats.users?.total || 0} color={COLORS.dk1} />
            <StatCard label="Active Users" value={stats.users?.active || 0} color={COLORS.positive} />
            <StatCard label="Total Predictions" value={stats.users?.total_predictions || 0} color={COLORS.dk2} />
            <StatCard
              label="Forecasts (24h)"
              value={stats.forecasts_24h?.count || 0}
              sub={stats.forecasts_24h?.avg_confidence ? `Avg confidence: ${stats.forecasts_24h.avg_confidence}%` : ""}
              color={COLORS.accent}
            />
          </div>

          {/* Table counts */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, marginBottom: 10 }}>Database Tables</div>
            {stats.tables && Object.entries(stats.tables).map(([table, count]) => (
              <div key={table} style={{
                display: "flex", justifyContent: "space-between", padding: "8px 12px",
                borderBottom: `1px solid ${COLORS.border}`,
              }}>
                <span style={{ fontSize: 12, color: COLORS.textMuted }}>{table.replace(/_/g, " ")}</span>
                <span style={{ fontSize: 12, fontWeight: 700, color: COLORS.text, fontVariantNumeric: "tabular-nums" }}>
                  {count.toLocaleString()}
                </span>
              </div>
            ))}
          </div>

          {/* Data freshness */}
          {stats.data_range && (
            <div style={{ padding: "12px 14px", background: COLORS.bg, borderRadius: 8 }}>
              <div style={{ fontSize: 11, color: COLORS.textDim, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.06em" }}>Data Range</div>
              <div style={{ fontSize: 12, color: COLORS.textMuted }}>
                {stats.data_range.oldest} → {stats.data_range.newest}
              </div>
            </div>
          )}

          {/* Refresh button */}
          <button
            onClick={() => fetchData("stats")}
            style={{
              marginTop: 16, padding: "8px 20px", borderRadius: 8,
              border: `1px solid ${COLORS.border}`, background: COLORS.surfaceLight,
              color: COLORS.textMuted, fontSize: 12, fontWeight: 600, cursor: "pointer",
            }}
          >
            ↻ Refresh Stats
          </button>
        </>
      )}

      {/* ══════ USERS ══════ */}
      {activeSection === "users" && users && (
        <>
          <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
            <StatCard label="Total" value={users.total_users} color={COLORS.dk1} />
            <StatCard label="Active" value={users.active_users} color={COLORS.positive} />
            <StatCard label="Admins" value={users.admin_users} color={COLORS.warning} />
          </div>

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: `2px solid ${COLORS.border}` }}>
                  {["Email", "Role", "Predictions", "Last Login", "Status", "Action"].map((h) => (
                    <th key={h} style={{ padding: "10px 12px", textAlign: "left", color: COLORS.textDim, fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(users.users || []).map((u) => (
                  <tr key={u.uid || u.email} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                    <td style={{ padding: "10px 12px", color: COLORS.text }}>{u.email}</td>
                    <td style={{ padding: "10px 12px" }}>
                      <span style={{
                        fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 4,
                        color: u.role === "admin" ? COLORS.warning : COLORS.accent,
                        background: u.role === "admin" ? COLORS.warning + "18" : COLORS.accent + "18",
                        textTransform: "uppercase", letterSpacing: "0.06em",
                      }}>
                        {u.role}
                      </span>
                    </td>
                    <td style={{ padding: "10px 12px", color: COLORS.textMuted, fontVariantNumeric: "tabular-nums" }}>
                      {u.prediction_count || 0}
                    </td>
                    <td style={{ padding: "10px 12px", color: COLORS.textDim, fontSize: 11 }}>
                      {u.last_login ? new Date(u.last_login).toLocaleString("da-DK") : "—"}
                    </td>
                    <td style={{ padding: "10px 12px" }}>
                      <span style={{
                        width: 8, height: 8, borderRadius: "50%", display: "inline-block",
                        background: u.is_active !== false ? COLORS.positive : COLORS.negative,
                      }} />
                    </td>
                    <td style={{ padding: "10px 12px" }}>
                      <button
                        onClick={() => toggleUser(u.uid, u.is_active !== false)}
                        style={{
                          padding: "4px 12px", borderRadius: 4, border: "none",
                          background: u.is_active !== false ? COLORS.negative + "20" : COLORS.positive + "20",
                          color: u.is_active !== false ? COLORS.negative : COLORS.positive,
                          fontSize: 10, fontWeight: 700, cursor: "pointer",
                          textTransform: "uppercase", letterSpacing: "0.04em",
                        }}
                      >
                        {u.is_active !== false ? "Deactivate" : "Activate"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ══════ FORECAST LOGS ══════ */}
      {activeSection === "logs" && logs && (
        <>
          <div style={{ fontSize: 12, color: COLORS.textDim, marginBottom: 12 }}>
            Showing {logs.count} entries from last {logs.hours}h
          </div>

          <div style={{ overflowX: "auto", maxHeight: 500, overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead style={{ position: "sticky", top: 0, background: COLORS.surface }}>
                <tr style={{ borderBottom: `2px solid ${COLORS.border}` }}>
                  {["Time", "Zone", "Hour", "Predicted", "Actual", "Confidence", "Models"].map((h) => (
                    <th key={h} style={{ padding: "8px 10px", textAlign: "left", color: COLORS.textDim, fontWeight: 600, fontSize: 10, textTransform: "uppercase" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(logs.logs || []).map((log, i) => (
                  <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                    <td style={{ padding: "6px 10px", color: COLORS.textDim, fontVariantNumeric: "tabular-nums" }}>
                      {log.created_at ? new Date(log.created_at).toLocaleString("da-DK", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" }) : "—"}
                    </td>
                    <td style={{ padding: "6px 10px", color: COLORS.dk1, fontWeight: 600 }}>{log.zone}</td>
                    <td style={{ padding: "6px 10px", color: COLORS.textMuted }}>{log.forecast_hour}</td>
                    <td style={{ padding: "6px 10px", color: COLORS.text, fontVariantNumeric: "tabular-nums" }}>
                      €{log.predicted_price?.toFixed(2) ?? "—"}
                    </td>
                    <td style={{ padding: "6px 10px", color: log.actual_price ? COLORS.positive : COLORS.textDim, fontVariantNumeric: "tabular-nums" }}>
                      {log.actual_price ? `€${log.actual_price.toFixed(2)}` : "pending"}
                    </td>
                    <td style={{ padding: "6px 10px", color: COLORS.textMuted }}>{log.confidence?.toFixed(1) ?? "—"}%</td>
                    <td style={{ padding: "6px 10px", color: COLORS.textDim }}>{log.models_used ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
