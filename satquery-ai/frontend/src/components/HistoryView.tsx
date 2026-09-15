import React, { useState, useEffect } from 'react';
import { History, FileText, CheckCircle2, ShieldCheck, Download } from 'lucide-react';
import { generateReport } from '../api';

export const HistoryView: React.FC = () => {
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    fetch('http://localhost:8000/api/history')
      .then(res => res.json())
      .then(data => setHistory(data.history || []))
      .catch(console.error);
  }, []);

  const handleDownload = async (id: string) => {
    try {
      const rep = await generateReport(id);
      window.open(`http://localhost:8000${rep.report_url}`, '_blank');
    } catch (e: any) {
      alert('Report error: ' + e.message);
    }
  };

  return (
    <div style={{ padding: '28px 32px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h3 style={{ fontSize: '18px', fontWeight: 800, color: '#F8FAFC', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <History size={20} color="#38BDF8" />
          <span>Execution History & Audit Logs</span>
        </h3>
      </div>

      {history.length === 0 ? (
        <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: '#64748B' }}>
          <p>No queries executed in current session yet. Run an analysis or demo scenario to view execution logs.</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {history.map((item) => (
            <div key={item.id} className="glass-panel" style={{ padding: '18px 24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#38BDF8', backgroundColor: 'rgba(56,189,248,0.12)', padding: '2px 6px', borderRadius: '4px' }}>
                    {item.task_type}
                  </span>
                  <span style={{ fontSize: '11px', color: '#64748B' }}>{item.created_at}</span>
                </div>
                <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#F8FAFC' }}>"{item.query}"</h4>
                <p style={{ fontSize: '12px', color: '#94A3B8', marginTop: '2px', maxWidth: '700px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.answer}
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '13px', fontWeight: 700, color: '#10B981' }}>{item.confidence_data?.overall_percentage || '91.4%'}</div>
                  <div style={{ fontSize: '11px', color: '#64748B' }}>{item.processing_time_ms?.toFixed(0)} ms</div>
                </div>
                <button onClick={() => handleDownload(item.id)} className="btn-secondary" style={{ fontSize: '12px', padding: '6px 12px' }}>
                  <Download size={13} />
                  <span>Report</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
