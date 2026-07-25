import React, { useState, useEffect } from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || '';

const MetricCard = ({ label, value, unit, status, subtitle }) => {
  const statusColors = {
    good: '#22c55e',
    warning: '#eab308',
    bad: '#ef4444',
    neutral: '#94a3b8'
  };

  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: '8px',
      padding: '14px 16px',
      minWidth: '120px',
      flex: 1
    }}>
      <div style={{
        fontSize: '11px',
        color: '#94a3b8',
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
        marginBottom: '6px'
      }}>
        {label}
      </div>
      <div style={{
        fontSize: '22px',
        fontWeight: 600,
        color: statusColors[status] || '#e2e8f0',
        fontVariantNumeric: 'tabular-nums'
      }}>
        {value !== null && value !== undefined ? `${value}${unit || ''}` : '—'}
      </div>
      {subtitle && (
        <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
          {subtitle}
        </div>
      )}
    </div>
  );
};

const ModelRow = ({ model, rank }) => {
  const medals = ['🥇', '🥈', '🥉'];
  const medal = rank <= 3 ? medals[rank - 1] : `#${rank}`;

  if (model.frozen) {
    return (
      <tr style={{ opacity: 0.4 }}>
        <td style={cellStyle}>{medal}</td>
        <td style={cellStyle}>{formatModelName(model.model_name)}</td>
        <td style={{ ...cellStyle, color: '#ef4444' }}>⚠️ Frozen</td>
        <td style={cellStyle}>—</td>
        <td style={cellStyle}>—</td>
      </tr>
    );
  }

  return (
    <tr>
      <td style={cellStyle}>{medal}</td>
      <td style={{ ...cellStyle, color: '#e2e8f0' }}>{formatModelName(model.model_name)}</td>
      <td style={cellStyle}>€{model.mae ?? '—'}</td>
      <td style={cellStyle}>{model.mape != null ? `${model.mape}%` : '—'}</td>
      <td style={{
        ...cellStyle,
        color: (model.directional_accuracy ?? 0) >= 0.7 ? '#22c55e' : '#eab308'
      }}>
        {model.directional_accuracy != null
          ? `${(model.directional_accuracy * 100).toFixed(0)}%`
          : '—'}
      </td>
    </tr>
  );
};

const cellStyle = {
  padding: '8px 12px',
  fontSize: '13px',
  color: '#94a3b8',
  borderBottom: '1px solid rgba(255,255,255,0.05)',
  textAlign: 'left'
};

const headerCellStyle = {
  ...cellStyle,
  fontSize: '11px',
  color: '#64748b',
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
  fontWeight: 600
};

function formatModelName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace('Cnn Lstm', 'CNN-LSTM')
    .replace('Lstm Gru', 'LSTM-GRU')
    .replace('N Beats', 'N-BEATS')
    .replace('Tcn', 'TCN')
    .replace('Xgboost', 'XGBoost')
    .replace('Sklearn', 'Sklearn');
}

function getMetricStatus(metric, value) {
  if (value === null || value === undefined) return 'neutral';
  switch (metric) {
    case 'mae':
      return value < 3 ? 'good' : value < 6 ? 'warning' : 'bad';
    case 'mape':
      return value < 5 ? 'good' : value < 10 ? 'warning' : 'bad';
    case 'directional':
      return value >= 0.75 ? 'good' : value >= 0.6 ? 'warning' : 'bad';
    default:
      return 'neutral';
  }
}

function formatHour(isoStr) {
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return isoStr;
  }
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;

  const predicted = payload.find(p => p.dataKey === 'predicted');
  const actual = payload.find(p => p.dataKey === 'actual');

  return (
    <div style={{
      background: '#1e293b',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '6px',
      padding: '10px 14px',
      fontSize: '12px'
    }}>
      <div style={{ color: '#94a3b8', marginBottom: '6px' }}>{label}</div>
      {predicted && (
        <div style={{ color: '#60a5fa' }}>
          Predicted: €{predicted.value?.toFixed(2)}
        </div>
      )}
      {actual && (
        <div style={{ color: '#22c55e' }}>
          Actual: €{actual.value?.toFixed(2)}
        </div>
      )}
      {predicted && actual && (
        <div style={{
          color: Math.abs(predicted.value - actual.value) < 3 ? '#22c55e' : '#ef4444',
          marginTop: '4px',
          borderTop: '1px solid rgba(255,255,255,0.1)',
          paddingTop: '4px'
        }}>
          Error: €{Math.abs(predicted.value - actual.value).toFixed(2)}
        </div>
      )}
    </div>
  );
};


export default function AccuracyTracker({ zone = 'DK1' }) {
  const [data, setData] = useState(null);
  const [hours, setHours] = useState(720);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAccuracy();
  }, [zone, hours]);

  async function fetchAccuracy() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/accuracy/latest?zone=${zone}&hours=${hours}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div style={{ color: '#94a3b8', padding: '40px', textAlign: 'center' }}>
        Loading accuracy data...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ color: '#ef4444', padding: '20px' }}>
        <span style={{ marginRight: '8px' }}>⚠️</span>
        {error === 'HTTP 500'
          ? 'No forecast data logged yet. Accuracy tracking starts after the first forecast cycle.'
          : `Error: ${error}`}
      </div>
    );
  }

  if (!data || !data.metrics || data.metrics.n_predictions === 0) {
    return (
      <div style={{
        color: '#94a3b8',
        padding: '40px',
        textAlign: 'center',
        background: 'rgba(255,255,255,0.02)',
        borderRadius: '8px'
      }}>
        <div style={{ fontSize: '24px', marginBottom: '12px' }}>📊</div>
        <div style={{ fontWeight: 500, marginBottom: '8px' }}>No accuracy data yet</div>
        <div style={{ fontSize: '13px', color: '#64748b' }}>
          Forecasts are being logged. Accuracy metrics will appear once
          actual prices arrive for the predicted hours.
        </div>
      </div>
    );
  }

  const { metrics, pairs, per_model } = data;
  const chartData = pairs.map(p => ({
    ...p,
    hour: formatHour(p.hour)
  }));

  // Sort models by MAE for the table
  const modelEntries = Object.entries(per_model || {})
    .map(([name, m]) => ({ model_name: name, ...m }))
    .sort((a, b) => (a.frozen ? 1 : 0) - (b.frozen ? 1 : 0) || (a.mae ?? 999) - (b.mae ?? 999));

  return (
    <div style={{ padding: '0' }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '16px'
      }}>
        <h3 style={{ color: '#e2e8f0', margin: 0, fontSize: '16px', fontWeight: 600 }}>
          Forecast Accuracy
        </h3>
        <div style={{ display: 'flex', gap: '6px' }}>
          {[24, 48, 72].map(h => (
            <button
              key={h}
              onClick={() => setHours(h)}
              style={{
                padding: '4px 12px',
                fontSize: '12px',
                borderRadius: '4px',
                border: '1px solid',
                borderColor: hours === h ? '#3b82f6' : 'rgba(255,255,255,0.1)',
                background: hours === h ? 'rgba(59,130,246,0.15)' : 'transparent',
                color: hours === h ? '#60a5fa' : '#94a3b8',
                cursor: 'pointer'
              }}
            >
              {h}h
            </button>
          ))}
        </div>
      </div>

      {/* Metric Cards */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' }}>
        <MetricCard
          label="MAE"
          value={metrics.mae}
          unit=" €"
          status={getMetricStatus('mae', metrics.mae)}
          subtitle="Avg absolute error"
        />
        <MetricCard
          label="MAPE"
          value={metrics.mape}
          unit="%"
          status={getMetricStatus('mape', metrics.mape)}
          subtitle="Avg % error"
        />
        <MetricCard
          label="Direction"
          value={metrics.directional_accuracy != null
            ? (metrics.directional_accuracy * 100).toFixed(0)
            : null}
          unit="%"
          status={getMetricStatus('directional', metrics.directional_accuracy)}
          subtitle="Up/down correct"
        />
        <MetricCard
          label="Bias"
          value={metrics.bias}
          unit=" €"
          status="neutral"
          subtitle={metrics.bias > 0 ? 'Over-predicting' : metrics.bias < 0 ? 'Under-predicting' : 'Balanced'}
        />
        <MetricCard
          label="Predictions"
          value={metrics.n_predictions}
          status="neutral"
          subtitle={`Last ${hours}h`}
        />
      </div>

      {/* Predicted vs Actual Chart */}
      {chartData.length > 0 && (
        <div style={{
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '20px'
        }}>
          <div style={{
            fontSize: '12px',
            color: '#64748b',
            marginBottom: '12px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}>
            Predicted vs Actual
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="hour"
                tick={{ fill: '#64748b', fontSize: 11 }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => `€${v}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                dataKey="actual"
                fill="rgba(34,197,94,0.08)"
                stroke="none"
              />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#22c55e"
                strokeWidth={2}
                dot={false}
                name="Actual"
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#60a5fa"
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={false}
                name="Predicted"
              />
            </ComposedChart>
          </ResponsiveContainer>
          <div style={{
            display: 'flex',
            gap: '20px',
            justifyContent: 'center',
            marginTop: '8px',
            fontSize: '12px'
          }}>
            <span style={{ color: '#22c55e' }}>── Actual</span>
            <span style={{ color: '#60a5fa' }}>- - Predicted</span>
          </div>
        </div>
      )}

      {/* Per-Model Accuracy Table */}
      {modelEntries.length > 0 && (
        <div style={{
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '8px',
          overflow: 'hidden'
        }}>
          <div style={{
            fontSize: '12px',
            color: '#64748b',
            padding: '12px 16px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            borderBottom: '1px solid rgba(255,255,255,0.05)'
          }}>
            Per-Model Accuracy
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={headerCellStyle}>Rank</th>
                <th style={headerCellStyle}>Model</th>
                <th style={headerCellStyle}>MAE</th>
                <th style={headerCellStyle}>MAPE</th>
                <th style={headerCellStyle}>Direction</th>
              </tr>
            </thead>
            <tbody>
              {modelEntries.map((model, i) => (
                <ModelRow key={model.model_name} model={model} rank={i + 1} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
