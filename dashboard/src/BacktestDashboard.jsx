import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || '';

const MODEL_COLORS = {
  ensemble: '#f59e0b',
  advanced_transformer: '#3b82f6',
  cnn_lstm: '#ef4444',
  enhanced_tcn: '#22c55e',
  enhanced_informer: '#a855f7',
  lstm_gru: '#ec4899',
  enhanced_n_beats: '#06b6d4',
  xgboost: '#f97316',
  sklearn_ensemble: '#84cc16'
};

function formatDate(dateStr) {
  try {
    const d = new Date(dateStr + 'T00:00:00');
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } catch {
    return dateStr;
  }
}

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

const DayCard = ({ label, day, type }) => {
  if (!day) return null;
  const color = type === 'best' ? '#22c55e' : '#ef4444';
  return (
    <div style={{
      background: 'rgba(255,255,255,0.03)',
      border: `1px solid ${color}22`,
      borderRadius: '6px',
      padding: '12px 14px',
      flex: 1
    }}>
      <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>{label}</div>
      <div style={{ fontSize: '14px', color, fontWeight: 600 }}>
        €{day.mae} MAE
      </div>
      <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
        {formatDate(day.date)}
      </div>
    </div>
  );
};

const MaeTrendTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: '#1e293b',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '6px',
      padding: '10px 14px',
      fontSize: '12px'
    }}>
      <div style={{ color: '#94a3b8', marginBottom: '4px' }}>{label}</div>
      <div style={{ color: '#60a5fa' }}>MAE: €{d?.mae}</div>
      {d?.mape != null && <div style={{ color: '#a855f7' }}>MAPE: {d.mape}%</div>}
      {d?.avg_price != null && <div style={{ color: '#94a3b8' }}>Avg Price: €{d.avg_price}</div>}
      {d?.price_volatility != null && (
        <div style={{ color: '#94a3b8' }}>Volatility: €{d.price_volatility}</div>
      )}
      {d?.n_predictions != null && (
        <div style={{ color: '#64748b', marginTop: '4px' }}>{d.n_predictions} predictions</div>
      )}
    </div>
  );
};


export default function BacktestDashboard({ zone = 'DK1' }) {
  const [historyData, setHistoryData] = useState(null);
  const [leaderboard, setLeaderboard] = useState(null);
  const [days, setDays] = useState(30);
  const [visibleModels, setVisibleModels] = useState(new Set(['ensemble']));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, [zone, days]);

  async function fetchData() {
    setLoading(true);
    setError(null);
    try {
      const [histRes, modelRes] = await Promise.all([
        fetch(`${API_BASE}/api/accuracy/history?zone=${zone}&days=${days}`),
        fetch(`${API_BASE}/api/accuracy/models?zone=${zone}&days=${days}`)
      ]);

      if (!histRes.ok) throw new Error(`History: HTTP ${histRes.status}`);
      if (!modelRes.ok) throw new Error(`Models: HTTP ${modelRes.status}`);

      const histJson = await histRes.json();
      const modelJson = await modelRes.json();

      setHistoryData(histJson);
      setLeaderboard(modelJson.leaderboard || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function toggleModel(name) {
    setVisibleModels(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  if (loading) {
    return (
      <div style={{ color: '#94a3b8', padding: '40px', textAlign: 'center' }}>
        Loading backtest data...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ color: '#ef4444', padding: '20px' }}>
        <span style={{ marginRight: '8px' }}>⚠️</span>
        {error.includes('500')
          ? 'No historical data yet. The backtest dashboard populates as forecast accuracy accumulates.'
          : `Error: ${error}`}
      </div>
    );
  }

  const hasData = historyData?.daily_metrics?.length > 0;

  if (!hasData) {
    return (
      <div style={{
        color: '#94a3b8',
        padding: '40px',
        textAlign: 'center',
        background: 'rgba(255,255,255,0.02)',
        borderRadius: '8px'
      }}>
        <div style={{ fontSize: '24px', marginBottom: '12px' }}>📈</div>
        <div style={{ fontWeight: 500, marginBottom: '8px' }}>No backtest data yet</div>
        <div style={{ fontSize: '13px', color: '#64748b' }}>
          Historical accuracy will appear after forecasts are logged and actuals arrive.
          Check back after 24-48 hours of forecast cycling.
        </div>
      </div>
    );
  }

  const { daily_metrics, per_model_daily, summary, error_distribution } = historyData;
  const chartData = daily_metrics.map(d => ({
    ...d,
    date: formatDate(d.date)
  }));

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
          Historical Backtest
        </h3>
        <div style={{ display: 'flex', gap: '6px' }}>
          {[7, 30, 90].map(d => (
            <button
              key={d}
              onClick={() => setDays(d)}
              style={{
                padding: '4px 12px',
                fontSize: '12px',
                borderRadius: '4px',
                border: '1px solid',
                borderColor: days === d ? '#3b82f6' : 'rgba(255,255,255,0.1)',
                background: days === d ? 'rgba(59,130,246,0.15)' : 'transparent',
                color: days === d ? '#60a5fa' : '#94a3b8',
                cursor: 'pointer'
              }}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>

      {/* Summary Strip */}
      {summary && (
        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px', flexWrap: 'wrap' }}>
          <div style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '8px',
            padding: '14px 16px',
            flex: 1.5
          }}>
            <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '6px' }}>
              OVERALL MAE
            </div>
            <div style={{
              fontSize: '26px',
              fontWeight: 700,
              color: summary.overall_mae < 5 ? '#22c55e' : '#eab308',
              fontVariantNumeric: 'tabular-nums'
            }}>
              €{summary.overall_mae}
            </div>
            <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
              {summary.days_below_target}/{summary.total_days} days below €{summary.target_mae} target
            </div>
          </div>
          <DayCard label="Best Day" day={summary.best_day} type="best" />
          <DayCard label="Worst Day" day={summary.worst_day} type="worst" />
        </div>
      )}

      {/* MAE Trend Chart */}
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
          Daily MAE Trend
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#64748b', fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: '#64748b', fontSize: 11 }}
              tickFormatter={v => `€${v}`}
            />
            <Tooltip content={<MaeTrendTooltip />} />
            <ReferenceLine
              y={summary?.target_mae || 5}
              stroke="#ef4444"
              strokeDasharray="4 4"
              strokeOpacity={0.5}
              label={{
                value: `Target €${summary?.target_mae || 5}`,
                fill: '#ef444488',
                fontSize: 10,
                position: 'right'
              }}
            />
            <Line
              type="monotone"
              dataKey="mae"
              stroke="#60a5fa"
              strokeWidth={2}
              dot={{ fill: '#60a5fa', r: 3 }}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Model Performance Over Time */}
      {per_model_daily && Object.keys(per_model_daily).length > 0 && (
        <div style={{
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '20px'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: '12px'
          }}>
            <div style={{
              fontSize: '12px',
              color: '#64748b',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              Model Comparison
            </div>
          </div>

          {/* Model toggle chips */}
          <div style={{
            display: 'flex',
            gap: '6px',
            flexWrap: 'wrap',
            marginBottom: '12px'
          }}>
            {Object.keys(per_model_daily).map(name => {
              const active = visibleModels.has(name);
              const color = MODEL_COLORS[name] || '#94a3b8';
              return (
                <button
                  key={name}
                  onClick={() => toggleModel(name)}
                  style={{
                    padding: '3px 10px',
                    fontSize: '11px',
                    borderRadius: '12px',
                    border: `1px solid ${active ? color : 'rgba(255,255,255,0.1)'}`,
                    background: active ? `${color}22` : 'transparent',
                    color: active ? color : '#64748b',
                    cursor: 'pointer'
                  }}
                >
                  {formatModelName(name)}
                </button>
              );
            })}
          </div>

          {/* Multi-model chart */}
          {(() => {
            // Merge all model daily data into one array keyed by date
            const dateMap = {};
            for (const [model, entries] of Object.entries(per_model_daily)) {
              if (!visibleModels.has(model)) continue;
              for (const e of entries) {
                const dateKey = e.date;
                if (!dateMap[dateKey]) dateMap[dateKey] = { date: formatDate(dateKey) };
                dateMap[dateKey][model] = e.mae;
              }
            }
            const merged = Object.values(dateMap).sort((a, b) =>
              a.date.localeCompare(b.date));

            if (merged.length === 0) {
              return (
                <div style={{ color: '#64748b', fontSize: '13px', padding: '20px', textAlign: 'center' }}>
                  Select models above to compare
                </div>
              );
            }

            return (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={merged} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} interval="preserveStartEnd" />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={v => `€${v}`} />
                  <Tooltip
                    contentStyle={{
                      background: '#1e293b',
                      border: '1px solid rgba(255,255,255,0.1)',
                      borderRadius: '6px',
                      fontSize: '12px'
                    }}
                    formatter={(val, name) => [`€${val}`, formatModelName(name)]}
                  />
                  {[...visibleModels].map(model => (
                    <Line
                      key={model}
                      type="monotone"
                      dataKey={model}
                      stroke={MODEL_COLORS[model] || '#94a3b8'}
                      strokeWidth={model === 'ensemble' ? 2.5 : 1.5}
                      dot={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            );
          })()}
        </div>
      )}

      {/* Error Distribution */}
      {error_distribution && error_distribution.length > 0 && (
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
            Error Distribution
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={error_distribution} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="bucket"
                tick={{ fill: '#64748b', fontSize: 10 }}
                label={{ value: '€ Error Range', fill: '#64748b', fontSize: 11, position: 'bottom' }}
              />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  background: '#1e293b',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: '6px',
                  fontSize: '12px'
                }}
                formatter={(val) => [`${val} hours`, 'Count']}
                labelFormatter={(label) => `€${label} error`}
              />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {error_distribution.map((entry, i) => {
                  const bucket = parseInt(entry.bucket.split('-')[0]);
                  const color = bucket < 3 ? '#22c55e' : bucket < 6 ? '#eab308' : '#ef4444';
                  return <Cell key={i} fill={color} fillOpacity={0.7} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model Leaderboard Table */}
      {leaderboard && leaderboard.length > 0 && (
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
            Model Leaderboard — {days} Day Cumulative
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Rank', 'Model', 'MAE', 'MAPE', 'RMSE', 'Direction', 'Predictions'].map(h => (
                  <th key={h} style={{
                    padding: '8px 12px',
                    fontSize: '11px',
                    color: '#64748b',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    fontWeight: 600,
                    textAlign: 'left',
                    borderBottom: '1px solid rgba(255,255,255,0.05)'
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {leaderboard.map((m, i) => {
                const medals = ['🥇', '🥈', '🥉'];
                const medal = i < 3 ? medals[i] : `#${i + 1}`;
                const isEnsemble = m.model_name === 'ensemble';
                return (
                  <tr key={m.model_name} style={{
                    opacity: m.frozen ? 0.4 : 1,
                    background: isEnsemble ? 'rgba(245,158,11,0.05)' : 'transparent'
                  }}>
                    <td style={tdStyle}>{medal}</td>
                    <td style={{
                      ...tdStyle,
                      color: isEnsemble ? '#f59e0b' : '#e2e8f0',
                      fontWeight: isEnsemble ? 600 : 400
                    }}>
                      {formatModelName(m.model_name)}
                      {m.frozen && <span style={{ color: '#ef4444', marginLeft: '6px' }}>⚠️</span>}
                    </td>
                    <td style={tdStyle}>{m.frozen ? '—' : `€${m.mae}`}</td>
                    <td style={tdStyle}>{m.frozen ? '—' : m.mape != null ? `${m.mape}%` : '—'}</td>
                    <td style={tdStyle}>{m.frozen ? '—' : m.rmse != null ? `€${m.rmse}` : '—'}</td>
                    <td style={{
                      ...tdStyle,
                      color: !m.frozen && (m.directional_accuracy ?? 0) >= 0.7 ? '#22c55e' : '#94a3b8'
                    }}>
                      {m.frozen ? '—' : m.directional_accuracy != null
                        ? `${(m.directional_accuracy * 100).toFixed(0)}%` : '—'}
                    </td>
                    <td style={tdStyle}>{m.n_predictions || 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const tdStyle = {
  padding: '8px 12px',
  fontSize: '13px',
  color: '#94a3b8',
  borderBottom: '1px solid rgba(255,255,255,0.05)',
  textAlign: 'left'
};
