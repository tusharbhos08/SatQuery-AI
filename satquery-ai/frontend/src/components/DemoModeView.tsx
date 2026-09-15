import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Play, 
  CheckCircle2, 
  ShieldCheck, 
  Layers, 
  Download, 
  Clock, 
  Zap, 
  ArrowRight,
  TrendingUp,
  Droplets,
  Search
} from 'lucide-react';
import { getDemoScenarios, executeQuery, generateReport } from '../api';
import { DemoScenario, AnalysisResult } from '../types';

export const DemoModeView: React.FC = () => {
  const [scenarios, setScenarios] = useState<DemoScenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<DemoScenario | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    getDemoScenarios().then((sc) => {
      setScenarios(sc);
      if (sc.length > 0) setSelectedScenario(sc[0]);
    }).catch(console.error);
  }, []);

  const handleRunDemo = async (scenario: DemoScenario) => {
    setSelectedScenario(scenario);
    setLoading(true);
    setResult(null);

    try {
      const res = await executeQuery(scenario.query, scenario.image_ids);
      setResult(res);
    } catch (e: any) {
      alert('Demo execution failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    try {
      const rep = await generateReport(result.id);
      window.open(`http://localhost:8000${rep.report_url}`, '_blank');
    } catch (e: any) {
      alert('Report download failed: ' + e.message);
    }
  };

  return (
    <div style={{ padding: '28px 32px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.15) 100%)',
        border: '1px solid rgba(6, 182, 212, 0.3)',
        borderRadius: '12px',
        padding: '24px 28px',
        marginBottom: '24px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', backgroundColor: '#06B6D4', color: '#000', padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 800, textTransform: 'uppercase', marginBottom: '8px' }}>
            SIH Demo Mode Active
          </div>
          <h2 style={{ fontSize: '22px', fontWeight: 800, color: '#F8FAFC', marginBottom: '4px' }}>
            Preloaded Remote Sensing Demonstration Scenarios
          </h2>
          <p style={{ fontSize: '13px', color: '#94A3B8' }}>
            Select any scenario below to trigger the live agentic orchestration and real algorithmic inference pipeline.
          </p>
        </div>
      </div>

      {/* Scenarios Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '14px', marginBottom: '28px' }}>
        {scenarios.map((sc) => {
          const isSelected = selectedScenario?.id === sc.id;
          return (
            <div
              key={sc.id}
              onClick={() => handleRunDemo(sc)}
              className="glass-card"
              style={{
                padding: '18px',
                cursor: 'pointer',
                borderColor: isSelected ? '#38BDF8' : 'rgba(255,255,255,0.06)',
                backgroundColor: isSelected ? 'rgba(56, 189, 248, 0.08)' : undefined
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#38BDF8', backgroundColor: 'rgba(56, 189, 248, 0.12)', padding: '2px 6px', borderRadius: '4px' }}>
                  DEMO {sc.id} • {sc.expected_task}
                </span>
                <Play size={14} color={isSelected ? '#38BDF8' : '#64748B'} />
              </div>
              <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#F8FAFC', marginBottom: '6px' }}>{sc.title}</h4>
              <p style={{ fontSize: '12px', color: '#94A3B8', lineHeight: 1.4, marginBottom: '10px' }}>{sc.description}</p>
              <div style={{ fontSize: '11.5px', color: '#60A5FA', fontStyle: 'italic' }}>"{sc.query}"</div>
            </div>
          );
        })}
      </div>

      {/* Execution Results View */}
      {loading && (
        <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>
          <div style={{ width: '40px', height: '40px', border: '3px solid #1E293B', borderTopColor: '#38BDF8', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 16px auto' }}></div>
          <h4 style={{ fontSize: '16px', fontWeight: 700, color: '#F8FAFC' }}>Executing Autonomous Agent Pipeline</h4>
          <p style={{ fontSize: '13px', color: '#94A3B8', marginTop: '4px' }}>Validating GeoTIFF • Routing query • Running specialist models • Estimating confidence</p>
        </div>
      )}

      {result && !loading && (
        <div className="glass-panel" style={{ padding: '28px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
            <div>
              <span style={{ fontSize: '11px', fontWeight: 700, color: '#10B981', textTransform: 'uppercase' }}>Workflow Execution Complete</span>
              <h3 style={{ fontSize: '18px', fontWeight: 800, color: '#F8FAFC', marginTop: '2px' }}>
                {selectedScenario?.title}
              </h3>
            </div>
            <button onClick={handleDownloadReport} className="btn-secondary" style={{ fontSize: '12px' }}>
              <Download size={14} />
              <span>Export Evaluation Report (PDF/HTML)</span>
            </button>
          </div>

          {/* AI Answer */}
          <div style={{ backgroundColor: '#0F172A', border: '1px solid #38BDF8', borderRadius: '8px', padding: '18px', fontSize: '14.5px', lineHeight: 1.6, marginBottom: '20px' }}>
            {result.answer}
          </div>

          {/* Dual Panel: Evidence + Trace */}
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px' }}>
            
            {/* Visual Evidence */}
            {result.visual_evidence?.primary_base64 && (
              <div>
                <h4 style={{ fontSize: '13px', fontWeight: 600, color: '#94A3B8', marginBottom: '8px' }}>
                  Visual Evidence Raster Overlay
                </h4>
                <div style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #334155' }}>
                  <img src={result.visual_evidence.primary_base64} alt="Evidence" style={{ width: '100%', height: 'auto', display: 'block' }} />
                </div>
              </div>
            )}

            {/* Observable Trace */}
            <div style={{ backgroundColor: '#0B0F19', borderRadius: '8px', padding: '16px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', marginBottom: '12px', display: 'flex', justifyContent: 'space-between' }}>
                <span>Observable Execution Trace</span>
                <span style={{ color: '#10B981' }}>{result.confidence_data.overall_percentage} Conf</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {result.execution_trace.steps.map((step, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                    <CheckCircle2 size={14} color="#10B981" />
                    <strong style={{ color: '#E2E8F0' }}>{step.step}</strong>
                    {step.time_ms > 0 && <span style={{ color: '#38BDF8', marginLeft: 'auto', fontSize: '11px' }}>{step.time_ms}ms</span>}
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
};
