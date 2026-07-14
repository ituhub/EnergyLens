/**
 * EnergyLens — Signal Quality Gate component.
 *
 * Displays a 5-gate forecast reliability assessment,
 * adapted from MarketLens for energy market forecasting.
 *
 * Usage:
 *   <SignalQualityGate qualityGate={forecastData.quality_gate} />
 *
 * Expected prop shape:
 *   {
 *     overall: "PASS" | "FAIL",
 *     passed: number,
 *     total: number,
 *     gates: [{ name, status, reason, icon }, ...],
 *     summary: string,
 *   }
 */

import { useState } from "react";

// ── Gate icons (inline SVG to avoid external deps) ──────────────────────

const GateIcons = {
  confidence: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 4.5V8.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="8" cy="11" r="0.75" fill="currentColor" />
    </svg>
  ),
  consensus: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path d="M3 8H13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M5 5H11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M5 11H11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
  freshness: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 4.5V8L10.5 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  stability: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <polyline points="2,11 5,7 8,9 11,4 14,6" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  volatility: (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <polyline points="2,8 4,4 6,12 8,6 10,10 12,3 14,8" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
};

// ── Status badge ────────────────────────────────────────────────────────

function StatusBadge({ status }) {
  const styles = {
    PASS: { background: "rgba(34, 197, 94, 0.15)", color: "#22c55e", border: "1px solid rgba(34, 197, 94, 0.3)" },
    FAIL: { background: "rgba(239, 68, 68, 0.15)", color: "#ef4444", border: "1px solid rgba(239, 68, 68, 0.3)" },
    WARN: { background: "rgba(234, 179, 8, 0.15)", color: "#eab308", border: "1px solid rgba(234, 179, 8, 0.3)" },
  };

  const s = styles[status] || styles.WARN;

  return (
    <span
      style={{
        ...s,
        padding: "2px 10px",
        borderRadius: 4,
        fontSize: 11,
        fontWeight: 600,
        letterSpacing: "0.05em",
        textTransform: "uppercase",
        display: "inline-block",
        minWidth: 42,
        textAlign: "center",
      }}
    >
      {status}
    </span>
  );
}

// ── Confidence zone bar ─────────────────────────────────────────────────

function ConfidenceBar({ confidence }) {
  // Map 40-88 range to 0-100% position
  const minConf = 40;
  const maxConf = 88;
  const pos = Math.max(0, Math.min(100, ((confidence - minConf) / (maxConf - minConf)) * 100));

  // Sweet spot: 55-80%
  const sweetStart = ((55 - minConf) / (maxConf - minConf)) * 100;
  const sweetEnd = ((80 - minConf) / (maxConf - minConf)) * 100;

  return (
    <div style={{ margin: "12px 0 4px" }}>
      <div
        style={{
          fontSize: 10,
          color: "#8b949e",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          marginBottom: 6,
          fontWeight: 500,
        }}
      >
        Confidence Zone
      </div>
      <div
        style={{
          position: "relative",
          height: 8,
          borderRadius: 4,
          background: "rgba(255,255,255,0.06)",
          overflow: "visible",
        }}
      >
        {/* Sweet spot highlight */}
        <div
          style={{
            position: "absolute",
            left: `${sweetStart}%`,
            width: `${sweetEnd - sweetStart}%`,
            height: "100%",
            background: "rgba(34, 197, 94, 0.25)",
            borderRadius: 4,
          }}
        />
        {/* Marker */}
        <div
          style={{
            position: "absolute",
            left: `${pos}%`,
            top: -3,
            width: 3,
            height: 14,
            background: confidence >= 55 && confidence <= 80 ? "#22c55e" : "#eab308",
            borderRadius: 2,
            transform: "translateX(-1px)",
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 10,
          color: "#6e7681",
          marginTop: 4,
        }}
      >
        <span>{minConf}%</span>
        <span style={{ color: "#22c55e" }}>55-80% optimal</span>
        <span>{maxConf}%</span>
      </div>
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────

export default function SignalQualityGate({ qualityGate, confidence }) {
  const [expanded, setExpanded] = useState(true);

  if (!qualityGate) return null;

  const { overall, passed, total, gates, summary } = qualityGate;
  const isPass = overall === "PASS";

  return (
    <div
      style={{
        background: "#161b22",
        borderRadius: 12,
        border: "1px solid rgba(255,255,255,0.08)",
        padding: "20px 24px",
        marginTop: 16,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          cursor: "pointer",
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 18 }}>🛡️</span>
          <span
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: "#e6edf3",
              letterSpacing: "0.01em",
            }}
          >
            Signal Quality Gate
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: "#e6edf3",
            }}
          >
            {expanded ? "▾" : "▸"}
          </span>
        </div>
      </div>

      {expanded && (
        <>
          {/* Overall result banner */}
          <div
            style={{
              marginTop: 16,
              padding: "14px 18px",
              borderRadius: 8,
              background: isPass
                ? "rgba(34, 197, 94, 0.1)"
                : "rgba(239, 68, 68, 0.1)",
              border: `1px solid ${isPass ? "rgba(34, 197, 94, 0.25)" : "rgba(239, 68, 68, 0.25)"}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span
                style={{
                  fontSize: 16,
                  color: isPass ? "#22c55e" : "#ef4444",
                }}
              >
                {isPass ? "✓" : "✗"}
              </span>
              <span
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  color: isPass ? "#22c55e" : "#ef4444",
                  letterSpacing: "0.03em",
                }}
              >
                GATE: {overall}
              </span>
              <span
                style={{
                  fontSize: 13,
                  color: isPass ? "#22c55e" : "#ef4444",
                  opacity: 0.8,
                }}
              >
                ({passed}/{total})
              </span>
            </div>
          </div>

          {/* Confidence bar */}
          {confidence != null && <ConfidenceBar confidence={confidence} />}

          {/* Individual gates */}
          <div style={{ marginTop: 16 }}>
            {gates.map((gate, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 12,
                  padding: "10px 0",
                  borderBottom:
                    i < gates.length - 1
                      ? "1px solid rgba(255,255,255,0.05)"
                      : "none",
                }}
              >
                {/* Icon */}
                <span
                  style={{
                    color:
                      gate.status === "PASS"
                        ? "#22c55e"
                        : gate.status === "FAIL"
                        ? "#ef4444"
                        : "#eab308",
                    flexShrink: 0,
                    marginTop: 1,
                  }}
                >
                  {GateIcons[gate.icon] || GateIcons.confidence}
                </span>

                {/* Name */}
                <span
                  style={{
                    fontSize: 13,
                    color: "#e6edf3",
                    fontWeight: 500,
                    minWidth: 140,
                    flexShrink: 0,
                  }}
                >
                  {gate.name}
                </span>

                {/* Badge */}
                <span style={{ flexShrink: 0 }}>
                  <StatusBadge status={gate.status} />
                </span>

                {/* Reason */}
                <span
                  style={{
                    fontSize: 12,
                    color: "#8b949e",
                    lineHeight: 1.5,
                    flex: 1,
                  }}
                >
                  {gate.reason}
                </span>
              </div>
            ))}
          </div>

          {/* Summary */}
          <div
            style={{
              marginTop: 12,
              fontSize: 12,
              color: "#8b949e",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span>💡</span>
            <span>{summary}</span>
          </div>
        </>
      )}
    </div>
  );
}
