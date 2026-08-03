/**
 * EnergyLens — Prediction Landing Page.
 *
 * Shows after login, before the dashboard.
 * User clicks "Run Prediction" → animated 6-stage stepper plays →
 * dashboard appears with results.
 *
 * Props:
 *   user          — Firebase user object { displayName, email }
 *   token         — Firebase ID token for API auth
 *   onComplete()  — called when prediction finishes (transitions to dashboard)
 *   onLogout()    — sign out handler
 */

import { useState, useEffect, useRef } from "react";

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

const STAGES = [
  { id: "data",     icon: "📡", label: "Data Loading",       desc: "Spot prices · Weather · Generation" },
  { id: "features", icon: "⚙️", label: "Feature Engineering", desc: "125+ energy market features" },
  { id: "models",   icon: "🧠", label: "Model Loading",       desc: "7-model neural ensemble" },
  { id: "ensemble", icon: "📊", label: "Ensemble Forecast",   desc: "Multi-step price prediction" },
  { id: "gate",     icon: "🛡️", label: "Quality Gate",        desc: "5-check signal validation" },
  { id: "result",   icon: "✅", label: "Result",              desc: "Forecast ready" },
];

function StepperStage({ stage, status, duration, index }) {
  const isActive = status === "active";
  const isDone = status === "done";
  const isPending = status === "pending";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "12px 16px",
        borderRadius: 10,
        background: isActive
          ? COLORS.accent + "12"
          : isDone
          ? COLORS.positive + "08"
          : "transparent",
        border: `1px solid ${
          isActive ? COLORS.accent + "40" : isDone ? COLORS.positive + "20" : COLORS.border
        }`,
        transition: "all 0.4s ease",
        opacity: isPending ? 0.4 : 1,
      }}
    >
      {/* Step number / check */}
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: 8,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: isDone ? 16 : 14,
          fontWeight: 700,
          background: isDone
            ? COLORS.positive + "20"
            : isActive
            ? COLORS.accent + "20"
            : COLORS.surfaceLight,
          color: isDone ? COLORS.positive : isActive ? COLORS.accent : COLORS.textDim,
          transition: "all 0.3s",
        }}
      >
        {isDone ? "✓" : stage.icon}
      </div>

      {/* Label + description */}
      <div style={{ flex: 1 }}>
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: isDone ? COLORS.positive : isActive ? COLORS.accent : COLORS.textMuted,
            transition: "color 0.3s",
          }}
        >
          {stage.label}
        </div>
        <div style={{ fontSize: 11, color: COLORS.textDim, marginTop: 2 }}>{stage.desc}</div>
      </div>

      {/* Duration badge */}
      {isDone && duration > 0 && (
        <span
          style={{
            fontSize: 10,
            fontWeight: 600,
            color: COLORS.textDim,
            background: COLORS.surfaceLight,
            padding: "3px 8px",
            borderRadius: 6,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {duration < 1000 ? `${Math.round(duration)}ms` : `${(duration / 1000).toFixed(1)}s`}
        </span>
      )}

      {/* Active spinner */}
      {isActive && (
        <div
          style={{
            width: 16,
            height: 16,
            border: `2px solid ${COLORS.accent}30`,
            borderTopColor: COLORS.accent,
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
          }}
        />
      )}
    </div>
  );
}

export default function PredictionLanding({ user, token, onComplete, onLogout }) {
  const [phase, setPhase] = useState("idle"); // idle | running | done | error
  const [activeStage, setActiveStage] = useState(-1);
  const [stageDurations, setStageDurations] = useState({});
  const [traceData, setTraceData] = useState(null);
  const [error, setError] = useState(null);
  const timerRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const runPrediction = async () => {
    setPhase("running");
    setError(null);
    setActiveStage(0);

    // Animate through stages while waiting for API
    let currentStage = 0;
    const stageStartTimes = { 0: Date.now() };

    timerRef.current = setInterval(() => {
      // Advance stepper roughly every 500–900ms
      currentStage++;
      if (currentStage < STAGES.length - 1) {
        // Mark previous stage as done
        setStageDurations((prev) => ({
          ...prev,
          [currentStage - 1]: Date.now() - (stageStartTimes[currentStage - 1] || Date.now()),
        }));
        stageStartTimes[currentStage] = Date.now();
        setActiveStage(currentStage);
      }
    }, 600 + Math.random() * 400);

    try {
      const headers = { "Content-Type": "application/json" };
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      const res = await fetch(`${API_BASE}/forecast?zone=DK1&hours=24`, { headers });
      const data = await res.json();

      // Stop the animation timer
      clearInterval(timerRef.current);
      timerRef.current = null;

      if (data.error) {
        throw new Error(data.error);
      }

      // If API returned pipeline_trace, use real durations
      if (data.pipeline_trace) {
        setTraceData(data.pipeline_trace);
        const realDurations = {};
        data.pipeline_trace.stages.forEach((s, i) => {
          if (s.duration_ms > 0) realDurations[i] = s.duration_ms;
        });
        setStageDurations(realDurations);
      }

      // Complete all stages
      const finalDurations = {};
      STAGES.forEach((_, i) => {
        finalDurations[i] = stageDurations[i] || 200 + Math.random() * 300;
      });
      setStageDurations(finalDurations);
      setActiveStage(STAGES.length); // All done
      setPhase("done");

      // Brief pause to show completion, then transition
      setTimeout(() => {
        onComplete();
      }, 1200);
    } catch (err) {
      clearInterval(timerRef.current);
      timerRef.current = null;
      setError(err.message);
      setPhase("error");
    }
  };

  const getStageStatus = (index) => {
    if (index < activeStage) return "done";
    if (index === activeStage) return "active";
    return "pending";
  };

  const greeting = user?.displayName
    ? user.displayName.split(" ")[0]
    : user?.email?.split("@")[0] || "there";

  return (
    <div
      style={{
        background: COLORS.bg,
        minHeight: "100vh",
        fontFamily: "'Inter', -apple-system, sans-serif",
        color: COLORS.text,
      }}
    >
      {/* Top bar */}
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "14px 28px",
          borderBottom: `1px solid ${COLORS.border}`,
          background: COLORS.surface + "cc",
          backdropFilter: "blur(12px)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: `linear-gradient(135deg, ${COLORS.dk1}, ${COLORS.dk2})`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
              fontWeight: 800,
              color: COLORS.bg,
            }}
          >
            E
          </div>
          <span style={{ fontSize: 16, fontWeight: 800, letterSpacing: "-0.02em" }}>
            EnergyLens
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span style={{ fontSize: 12, color: COLORS.textDim }}>{user?.email}</span>
          <button
            onClick={onLogout}
            style={{
              padding: "5px 14px",
              borderRadius: 6,
              border: `1px solid ${COLORS.border}`,
              background: "transparent",
              color: COLORS.textMuted,
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Sign Out
          </button>
        </div>
      </header>

      {/* Main content */}
      <main
        style={{
          maxWidth: 600,
          margin: "0 auto",
          padding: "60px 28px",
        }}
      >
        {/* Greeting */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <h1
            style={{
              fontSize: 26,
              fontWeight: 800,
              margin: "0 0 8px",
              letterSpacing: "-0.03em",
            }}
          >
            {phase === "idle" && `Hey ${greeting} 👋`}
            {phase === "running" && "Running Forecast Pipeline..."}
            {phase === "done" && "Forecast Complete ✓"}
            {phase === "error" && "Pipeline Error"}
          </h1>
          <p style={{ fontSize: 14, color: COLORS.textMuted, margin: 0 }}>
            {phase === "idle" && "Nordic power market forecast — 7-model ensemble, 24-hour horizon"}
            {phase === "running" && "Processing DK1 zone with live market data"}
            {phase === "done" && "Transitioning to dashboard..."}
            {phase === "error" && "Something went wrong — see details below"}
          </p>
        </div>

        {/* Run Prediction Button (idle state) */}
        {phase === "idle" && (
          <div style={{ textAlign: "center", marginBottom: 40 }}>
            <button
              onClick={runPrediction}
              style={{
                padding: "16px 48px",
                borderRadius: 14,
                border: "none",
                background: `linear-gradient(135deg, ${COLORS.dk1}, ${COLORS.dk2})`,
                color: COLORS.bg,
                fontSize: 16,
                fontWeight: 800,
                cursor: "pointer",
                letterSpacing: "-0.01em",
                transition: "transform 0.2s, box-shadow 0.2s",
                boxShadow: `0 4px 24px ${COLORS.dk1}30`,
              }}
              onMouseOver={(e) => {
                e.target.style.transform = "scale(1.03)";
                e.target.style.boxShadow = `0 6px 32px ${COLORS.dk1}50`;
              }}
              onMouseOut={(e) => {
                e.target.style.transform = "scale(1)";
                e.target.style.boxShadow = `0 4px 24px ${COLORS.dk1}30`;
              }}
            >
              ⚡ Run Prediction
            </button>
          </div>
        )}

        {/* Stepper (running / done / error state) */}
        {phase !== "idle" && (
          <div
            style={{
              background: COLORS.surface,
              border: `1px solid ${COLORS.border}`,
              borderRadius: 14,
              padding: 20,
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            {STAGES.map((stage, i) => (
              <StepperStage
                key={stage.id}
                stage={stage}
                status={getStageStatus(i)}
                duration={stageDurations[i] || 0}
                index={i}
              />
            ))}

            {/* Total duration */}
            {phase === "done" && traceData && (
              <div
                style={{
                  marginTop: 8,
                  padding: "10px 16px",
                  background: COLORS.surfaceLight,
                  borderRadius: 8,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.positive }}>
                  Pipeline complete
                </span>
                <span
                  style={{
                    fontSize: 12,
                    color: COLORS.textMuted,
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  Total: {(traceData.total_duration_ms / 1000).toFixed(1)}s
                </span>
              </div>
            )}
          </div>
        )}

        {/* Error display */}
        {phase === "error" && error && (
          <div
            style={{
              marginTop: 16,
              background: "rgba(248, 113, 113, 0.08)",
              border: "1px solid rgba(248, 113, 113, 0.25)",
              borderRadius: 10,
              padding: "14px 18px",
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.negative, marginBottom: 6 }}>
              Error
            </div>
            <div style={{ fontSize: 12, color: COLORS.textMuted, lineHeight: 1.5 }}>{error}</div>
            <button
              onClick={() => {
                setPhase("idle");
                setActiveStage(-1);
                setStageDurations({});
                setError(null);
              }}
              style={{
                marginTop: 12,
                padding: "8px 20px",
                borderRadius: 8,
                border: `1px solid ${COLORS.border}`,
                background: COLORS.surfaceLight,
                color: COLORS.text,
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Try Again
            </button>
          </div>
        )}

        {/* Info cards (idle state) */}
        {phase === "idle" && (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            {[
              { icon: "🧠", title: "7 Models", desc: "Transformer, CNN-LSTM, TCN, Informer, LSTM-GRU, N-BEATS, XGBoost" },
              { icon: "📡", title: "3 Data Sources", desc: "Nord Pool spot prices, Open-Meteo weather, ENTSO-E generation" },
              { icon: "🛡️", title: "Quality Gate", desc: "5-check validation: confidence, consensus, freshness, stability, volatility" },
            ].map((card) => (
              <div
                key={card.title}
                style={{
                  flex: 1,
                  minWidth: 160,
                  background: COLORS.surface,
                  border: `1px solid ${COLORS.border}`,
                  borderRadius: 10,
                  padding: "16px 14px",
                }}
              >
                <div style={{ fontSize: 20, marginBottom: 8 }}>{card.icon}</div>
                <div style={{ fontSize: 13, fontWeight: 700, color: COLORS.text, marginBottom: 4 }}>
                  {card.title}
                </div>
                <div style={{ fontSize: 11, color: COLORS.textDim, lineHeight: 1.5 }}>
                  {card.desc}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Spinner keyframe animation */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
