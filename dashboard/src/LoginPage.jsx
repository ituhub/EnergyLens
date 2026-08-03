/**
 * EnergyLens — Login & Registration Page.
 *
 * Firebase Auth email/password flow.
 * On successful login, stores the Firebase user object and calls onLogin.
 * Firestore user doc is created server-side (api/auth.py) on first API call.
 *
 * Props:
 *   onLogin(user)  — called with Firebase user after successful auth
 *
 * Requires: firebase npm package
 *   npm install firebase
 */

import { useState } from "react";

// Firebase SDK — initialized in firebaseConfig.js (see bottom of file for setup)
import { auth } from "./firebaseConfig";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  updateProfile,
} from "firebase/auth";

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

export default function LoginPage({ onLogin }) {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      let userCredential;

      if (mode === "register") {
        userCredential = await createUserWithEmailAndPassword(auth, email, password);
        // Set display name
        if (name.trim()) {
          await updateProfile(userCredential.user, { displayName: name.trim() });
        }
      } else {
        userCredential = await signInWithEmailAndPassword(auth, email, password);
      }

      onLogin(userCredential.user);
    } catch (err) {
      // Map Firebase error codes to user-friendly messages
      const messages = {
        "auth/email-already-in-use": "This email is already registered — try signing in",
        "auth/invalid-email": "Please enter a valid email address",
        "auth/weak-password": "Password must be at least 6 characters",
        "auth/user-not-found": "No account found with this email",
        "auth/wrong-password": "Incorrect password",
        "auth/invalid-credential": "Invalid email or password",
        "auth/too-many-requests": "Too many attempts — please try again later",
      };
      setError(messages[err.code] || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        background: COLORS.bg,
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'Inter', -apple-system, sans-serif",
        padding: 20,
      }}
    >
      <div style={{ width: "100%", maxWidth: 420 }}>
        {/* Brand */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 14,
              background: `linear-gradient(135deg, ${COLORS.dk1}, ${COLORS.dk2})`,
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 28,
              fontWeight: 800,
              color: COLORS.bg,
              marginBottom: 16,
            }}
          >
            E
          </div>
          <h1
            style={{
              fontSize: 28,
              fontWeight: 800,
              color: COLORS.text,
              margin: "0 0 6px",
              letterSpacing: "-0.03em",
            }}
          >
            EnergyLens
          </h1>
          <p
            style={{
              fontSize: 13,
              color: COLORS.textDim,
              margin: 0,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              fontWeight: 500,
            }}
          >
            Nordic Power Market Intelligence
          </p>
        </div>

        {/* Card */}
        <div
          style={{
            background: COLORS.surface,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 16,
            padding: "32px 28px",
          }}
        >
          {/* Mode Toggle */}
          <div
            style={{
              display: "flex",
              gap: 4,
              background: COLORS.surfaceLight,
              borderRadius: 8,
              padding: 3,
              marginBottom: 24,
            }}
          >
            {[
              { id: "login", label: "Sign In" },
              { id: "register", label: "Create Account" },
            ].map((m) => (
              <button
                key={m.id}
                onClick={() => {
                  setMode(m.id);
                  setError(null);
                }}
                style={{
                  flex: 1,
                  padding: "8px 0",
                  fontSize: 13,
                  fontWeight: mode === m.id ? 700 : 500,
                  borderRadius: 6,
                  border: "none",
                  cursor: "pointer",
                  background: mode === m.id ? COLORS.accent + "22" : "transparent",
                  color: mode === m.id ? COLORS.accent : COLORS.textDim,
                  transition: "all 0.2s",
                }}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Form */}
          <div>
            {mode === "register" && (
              <div style={{ marginBottom: 16 }}>
                <label
                  style={{
                    display: "block",
                    fontSize: 12,
                    fontWeight: 600,
                    color: COLORS.textMuted,
                    marginBottom: 6,
                    letterSpacing: "0.04em",
                  }}
                >
                  Full Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your name"
                  style={{
                    width: "100%",
                    padding: "10px 14px",
                    background: COLORS.surfaceLight,
                    border: `1px solid ${COLORS.borderLight}`,
                    borderRadius: 8,
                    color: COLORS.text,
                    fontSize: 14,
                    outline: "none",
                    boxSizing: "border-box",
                  }}
                />
              </div>
            )}

            <div style={{ marginBottom: 16 }}>
              <label
                style={{
                  display: "block",
                  fontSize: 12,
                  fontWeight: 600,
                  color: COLORS.textMuted,
                  marginBottom: 6,
                  letterSpacing: "0.04em",
                }}
              >
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                style={{
                  width: "100%",
                  padding: "10px 14px",
                  background: COLORS.surfaceLight,
                  border: `1px solid ${COLORS.borderLight}`,
                  borderRadius: 8,
                  color: COLORS.text,
                  fontSize: 14,
                  outline: "none",
                  boxSizing: "border-box",
                }}
              />
            </div>

            <div style={{ marginBottom: 24 }}>
              <label
                style={{
                  display: "block",
                  fontSize: 12,
                  fontWeight: 600,
                  color: COLORS.textMuted,
                  marginBottom: 6,
                  letterSpacing: "0.04em",
                }}
              >
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={mode === "register" ? "Minimum 6 characters" : "Enter password"}
                autoComplete={mode === "register" ? "new-password" : "current-password"}
                style={{
                  width: "100%",
                  padding: "10px 14px",
                  background: COLORS.surfaceLight,
                  border: `1px solid ${COLORS.borderLight}`,
                  borderRadius: 8,
                  color: COLORS.text,
                  fontSize: 14,
                  outline: "none",
                  boxSizing: "border-box",
                }}
              />
            </div>

            {/* Error */}
            {error && (
              <div
                style={{
                  background: "rgba(248, 113, 113, 0.08)",
                  border: "1px solid rgba(248, 113, 113, 0.25)",
                  borderRadius: 8,
                  padding: "10px 14px",
                  marginBottom: 16,
                  fontSize: 12,
                  color: COLORS.negative,
                  fontWeight: 500,
                }}
              >
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              onClick={handleSubmit}
              disabled={loading || !email || !password}
              style={{
                width: "100%",
                padding: "12px 0",
                borderRadius: 10,
                border: "none",
                background: loading || !email || !password
                  ? COLORS.surfaceLight
                  : `linear-gradient(135deg, ${COLORS.dk1}, ${COLORS.dk2})`,
                color: loading || !email || !password ? COLORS.textDim : COLORS.bg,
                fontSize: 14,
                fontWeight: 700,
                cursor: loading || !email || !password ? "not-allowed" : "pointer",
                transition: "all 0.3s",
                letterSpacing: "-0.01em",
              }}
            >
              {loading
                ? "Authenticating..."
                : mode === "register"
                ? "Create Account"
                : "Sign In"}
            </button>
          </div>
        </div>

        {/* Footer */}
        <p
          style={{
            textAlign: "center",
            fontSize: 11,
            color: COLORS.textDim,
            marginTop: 24,
          }}
        >
          EnergyLens v0.5.0 — ITU Consultant
        </p>
      </div>
    </div>
  );
}
