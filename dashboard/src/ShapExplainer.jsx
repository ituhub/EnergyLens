import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie, ReferenceLine
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || '';

function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace('2m', '(2m)')
    .replace('10m', '(10m)')
    .replace('100m', '(100m)')
    .replace('6h', '(6h)')
    .replace('12h', '(12h)')
    .replace('24h', '(24h)')
    .replace('48h', '(48h)')
    .replace('1h', '(1h)');
}

const GROUP_COLORS = {
  'Price Lags': '#f59e0b',
  'Weather': '#3b82f6',
  'Calendar': '#a855f7',
  'Demand Patterns': '#22c55e',
  'Cross-Features': '#ec4899',
  'Other': '#64748b'
};

const GROUP_ICONS = {
  'Price Lags': '📊',
  'Weather': '🌤️',
  'Calendar': '📅',
  'Demand Patterns': '⚡',
  'Cross-Features': '🔗',
  'Other': '📌'
};

const WaterfallTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: '#1e293b',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '6px',
      padding: '10px 14px',
      fontSize: '12px',
      maxWidth: '240px'
    }}>
      <div style={{ color: '#e2e8f0', fontWeight: 500, marginBottom: '4px' }}>
        {formatFeatureName(d?.feature || '')}
      </div>
      <div style={{ color: '#94a3b8', marginBottom: '4px', fontSize: '11px' }}>
        Group: {d?.group}
      </div>
      <div style={{
        color: d?.shap_value > 0 ? '#ef4444' : '#22c55e',
        fontWeight: 600
      }}>
        {d?.shap_value > 0 ? '▲' : '▼'} €{Math.abs(d?.shap_value || 0).toFixed(3)}
      </div>
      {d?.input_value != null && (
        <div style={{ color: '#64748b', marginTop: '4px', fontSize: '11px' }}>
          Input value: {d.input_value}
        </div>
      )}
    </div>
  );
};


export default function ShapExplainer({ zone = 'DK1' }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    fetchShap();
  }, [zone]);

  async function fetchShap() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/explain?zone=${zone}`);
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
        Computing feature importance...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ color: '#ef4444', padding: '20px' }}>
        <span style={{ marginRight: '8px' }}>⚠️</span>Error: {error}
      </div>
    );
  }

  if (!data) return null;

  const {
    model_used, method, base_value, predicted_value,
    top_features, group_summary, note
  } = data;

  const features = showAll ? (data.all_features || top_features) : (top_features || []);

  // Prepare waterfall chart data (top 10)
  const waterfallData = features.slice(0, 10).map(f => ({
    feature: f.feature.length > 18 ? f.feature.slice(0, 16) + '…' : f.feature,
    fullName: f.feature,
    shap_value: f.shap_value ?? 0,
    abs_shap: f.abs_shap,
    group: f.group,
    direction: f.direction,
    input_value: f.input_value
  }));

  // Group data for pie chart
  const pieData = (group_summary || []).map(g => ({
    name: g.group,
    value: Math.round(g.total_impact * 1000) / 10,
    color: GROUP_COLORS[g.group] || '#64748b'
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
          Forecast Explainability
        </h3>
        <div style={{
          fontSize: '11px',
          color: '#64748b',
          background: 'rgba(255,255,255,0.04)',
          padding: '4px 10px',
          borderRadius: '4px'
        }}>
          {method === 'TreeSHAP' ? '🎯 SHAP' : method === 'feature_importance' ? '📊 Importance' : '📖 Domain'}
          {' · '}{model_used}
        </div>
      </div>

      {/* Note for fallback mode */}
      {note && (
        <div style={{
          background: 'rgba(245,158,11,0.08)',
          border: '1px solid rgba(245,158,11,0.2)',
          borderRadius: '6px',
          padding: '10px 14px',
          marginBottom: '16px',
          fontSize: '12px',
          color: '#fbbf24'
        }}>
          {note}
        </div>
      )}

      {/* Base → Predicted Strip (SHAP mode only) */}
      {base_value != null && predicted_value != null && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          marginBottom: '20px',
          padding: '14px 16px',
          background: 'rgba(255,255,255,0.03)',
          borderRadius: '8px'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>Base Value</div>
            <div style={{ fontSize: '20px', fontWeight: 600, color: '#94a3b8' }}>€{base_value}</div>
          </div>
          <div style={{ color: '#64748b', fontSize: '20px' }}>→</div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>Predicted</div>
            <div style={{
              fontSize: '20px',
              fontWeight: 600,
              color: predicted_value > base_value ? '#ef4444' : '#22c55e'
            }}>
              €{predicted_value}
            </div>
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>Net Effect</div>
            <div style={{
              fontSize: '16px',
              fontWeight: 600,
              color: (predicted_value - base_value) > 0 ? '#ef4444' : '#22c55e'
            }}>
              {(predicted_value - base_value) > 0 ? '+' : ''}€{(predicted_value - base_value).toFixed(2)}
            </div>
          </div>
        </div>
      )}

      {/* Feature Group Summary Cards */}
      {group_summary && group_summary.length > 0 && (
        <div style={{
          display: 'flex',
          gap: '10px',
          marginBottom: '20px',
          flexWrap: 'wrap'
        }}>
          {group_summary.map(g => (
            <div key={g.group} style={{
              background: 'rgba(255,255,255,0.03)',
              border: `1px solid ${GROUP_COLORS[g.group] || '#64748b'}22`,
              borderRadius: '8px',
              padding: '12px 14px',
              flex: 1,
              minWidth: '140px'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                marginBottom: '6px'
              }}>
                <span style={{ fontSize: '14px' }}>{GROUP_ICONS[g.group] || '📌'}</span>
                <span style={{
                  fontSize: '11px',
                  color: GROUP_COLORS[g.group] || '#94a3b8',
                  fontWeight: 600
                }}>
                  {g.group}
                </span>
              </div>
              <div style={{
                fontSize: '18px',
                fontWeight: 600,
                color: '#e2e8f0'
              }}>
                {(g.total_impact * 100).toFixed(0)}%
              </div>
              <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                {g.feature_count} features
                {g.net_direction && g.net_direction !== 'unknown' && (
                  <span style={{
                    color: g.net_direction === 'up' ? '#ef4444' : '#22c55e',
                    marginLeft: '6px'
                  }}>
                    {g.net_direction === 'up' ? '▲ pushing up' : '▼ pushing down'}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Waterfall Chart — Top Features */}
      {waterfallData.length > 0 && (
        <div style={{
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '8px',
          padding: '16px',
          marginBottom: '20px'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '12px'
          }}>
            <div style={{
              fontSize: '12px',
              color: '#64748b',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              Top Feature Contributions
            </div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>
              <span style={{ color: '#ef4444' }}>■</span> pushes price up
              {' · '}
              <span style={{ color: '#22c55e' }}>■</span> pushes price down
            </div>
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={waterfallData}
              layout="vertical"
              margin={{ top: 5, right: 30, bottom: 5, left: 120 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#64748b', fontSize: 11 }}
                tickFormatter={v => `€${v}`}
              />
              <YAxis
                type="category"
                dataKey="feature"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                width={110}
              />
              <Tooltip content={<WaterfallTooltip />} />
              <ReferenceLine x={0} stroke="rgba(255,255,255,0.1)" />
              <Bar dataKey="shap_value" radius={[0, 3, 3, 0]}>
                {waterfallData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.shap_value > 0 ? '#ef4444' : '#22c55e'}
                    fillOpacity={0.75}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Group Breakdown Pie */}
      {pieData.length > 0 && (
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
            Feature Group Impact
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <ResponsiveContainer width="50%" height={200}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  dataKey="value"
                  strokeWidth={0}
                >
                  {pieData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} fillOpacity={0.8} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: '#1e293b',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '6px',
                    fontSize: '12px'
                  }}
                  formatter={(val) => [`${val}%`, 'Impact']}
                />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ flex: 1 }}>
              {pieData.map(g => (
                <div key={g.name} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginBottom: '8px',
                  fontSize: '12px'
                }}>
                  <div style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '2px',
                    background: g.color,
                    flexShrink: 0
                  }} />
                  <span style={{ color: '#94a3b8' }}>{g.name}</span>
                  <span style={{ color: '#e2e8f0', fontWeight: 500, marginLeft: 'auto' }}>
                    {g.value}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Full Feature Table (expandable) */}
      {features.length > 0 && (
        <div style={{
          background: 'rgba(255,255,255,0.02)',
          borderRadius: '8px',
          overflow: 'hidden'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '12px 16px',
            borderBottom: '1px solid rgba(255,255,255,0.05)'
          }}>
            <div style={{
              fontSize: '12px',
              color: '#64748b',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              All Features ({data.n_features || features.length})
            </div>
            {data.all_features && data.all_features.length > 15 && (
              <button
                onClick={() => setShowAll(!showAll)}
                style={{
                  fontSize: '11px',
                  color: '#60a5fa',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                {showAll ? 'Show Top 15' : 'Show All'}
              </button>
            )}
          </div>
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Feature', 'Group', 'Impact', 'Direction'].map(h => (
                    <th key={h} style={{
                      padding: '8px 12px',
                      fontSize: '11px',
                      color: '#64748b',
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px',
                      fontWeight: 600,
                      textAlign: 'left',
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      position: 'sticky',
                      top: 0,
                      background: '#0f172a'
                    }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {features.map((f, i) => (
                  <tr key={f.feature} style={{
                    background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)'
                  }}>
                    <td style={{
                      padding: '6px 12px',
                      fontSize: '12px',
                      color: '#e2e8f0',
                      borderBottom: '1px solid rgba(255,255,255,0.03)'
                    }}>
                      {formatFeatureName(f.feature)}
                    </td>
                    <td style={{
                      padding: '6px 12px',
                      fontSize: '11px',
                      color: GROUP_COLORS[f.group] || '#64748b',
                      borderBottom: '1px solid rgba(255,255,255,0.03)'
                    }}>
                      {f.group}
                    </td>
                    <td style={{
                      padding: '6px 12px',
                      fontSize: '12px',
                      color: '#94a3b8',
                      fontVariantNumeric: 'tabular-nums',
                      borderBottom: '1px solid rgba(255,255,255,0.03)'
                    }}>
                      {f.abs_shap?.toFixed(4) || '—'}
                    </td>
                    <td style={{
                      padding: '6px 12px',
                      fontSize: '12px',
                      borderBottom: '1px solid rgba(255,255,255,0.03)'
                    }}>
                      {f.direction === 'up' && <span style={{ color: '#ef4444' }}>▲ Up</span>}
                      {f.direction === 'down' && <span style={{ color: '#22c55e' }}>▼ Down</span>}
                      {f.direction === 'neutral' && <span style={{ color: '#64748b' }}>— Neutral</span>}
                      {f.direction === 'unknown' && <span style={{ color: '#64748b' }}>—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
