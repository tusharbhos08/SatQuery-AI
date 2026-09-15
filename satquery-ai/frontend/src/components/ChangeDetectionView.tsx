import React, { useState } from 'react';
import { GitCompare, Upload, Sparkles, TrendingUp, Download, CheckCircle2 } from 'lucide-react';
import { uploadImage, executeQuery, generateReport } from '../api';
import { AnalysisResult, UploadedImage } from '../types';

export const ChangeDetectionView: React.FC = () => {
  const [imgA, setImgA] = useState<UploadedImage | null>(null);
  const [imgB, setImgB] = useState<UploadedImage | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [sliderPos, setSliderPos] = useState(50);

  const handleUpload = async (file: File, target: 'a' | 'b') => {
    try {
      const data = await uploadImage(file);
      if (target === 'a') setImgA(data.image);
      else setImgB(data.image);
    } catch (e: any) {
      alert('Upload failed: ' + e.message);
    }
  };

  const handleRunChange = async () => {
    if (!imgA || !imgB) {
      alert('Please upload both Before (T1) and After (T2) images');
      return;
    }

    setLoading(true);
    try {
      const res = await executeQuery("What changed between these two dates and where did the change occur?", [imgA.id, imgB.id]);
      setResult(res);
    } catch (e: any) {
      alert('Analysis failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '28px 32px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
        
        {/* Upload T1 (Before) */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#38BDF8', marginBottom: '12px' }}>
            Observation 1 — Baseline (Date T1)
          </h4>
          <label style={{
            border: '2px dashed #334155',
            borderRadius: '8px',
            padding: '24px 16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            cursor: 'pointer',
            backgroundColor: 'rgba(15, 23, 42, 0.5)'
          }}>
            <input type="file" onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0], 'a')} style={{ display: 'none' }} />
            <Upload size={24} color="#38BDF8" style={{ marginBottom: '6px' }} />
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>
              {imgA ? imgA.filename : 'Upload Baseline GeoTIFF'}
            </span>
          </label>
        </div>

        {/* Upload T2 (After) */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#F43F5E', marginBottom: '12px' }}>
            Observation 2 — Post-Event (Date T2)
          </h4>
          <label style={{
            border: '2px dashed #334155',
            borderRadius: '8px',
            padding: '24px 16px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            cursor: 'pointer',
            backgroundColor: 'rgba(15, 23, 42, 0.5)'
          }}>
            <input type="file" onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0], 'b')} style={{ display: 'none' }} />
            <Upload size={24} color="#F43F5E" style={{ marginBottom: '6px' }} />
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>
              {imgB ? imgB.filename : 'Upload Target GeoTIFF'}
            </span>
          </label>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginBottom: '28px' }}>
        <button 
          onClick={handleRunChange} 
          disabled={loading || !imgA || !imgB}
          className="btn-primary"
          style={{ padding: '12px 28px', fontSize: '14.5px', opacity: (!imgA || !imgB) ? 0.6 : 1 }}
        >
          <GitCompare size={18} />
          <span>Execute Bi-Temporal Change Detection</span>
        </button>
      </div>

      {result && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#F8FAFC' }}>Bi-Temporal Change Assessment</h3>
            <span style={{ fontSize: '12.5px', fontWeight: 700, color: '#10B981', backgroundColor: 'rgba(16, 185, 129, 0.15)', padding: '4px 10px', borderRadius: '6px' }}>
              Confidence: {result.confidence_data.overall_percentage}
            </span>
          </div>

          <div style={{ backgroundColor: '#0F172A', border: '1px solid #38BDF8', borderRadius: '8px', padding: '16px', fontSize: '14px', lineHeight: 1.6, marginBottom: '20px' }}>
            {result.answer}
          </div>

          {/* Change Map & Side-by-Side */}
          {result.visual_evidence?.primary_base64 && (
            <div style={{ marginBottom: '24px' }}>
              <h4 style={{ fontSize: '13px', fontWeight: 600, color: '#94A3B8', marginBottom: '10px' }}>
                Categorical Change Map Visualization
              </h4>
              <img
                src={result.visual_evidence.primary_base64}
                alt="Change Map"
                style={{ width: '100%', maxHeight: '420px', objectFit: 'contain', borderRadius: '8px', border: '1px solid #334155', backgroundColor: '#0B0F19' }}
              />
            </div>
          )}

          {/* Statistics Grid */}
          {result.specific_metrics?.change_statistics && (
            <div style={{ backgroundColor: '#0B0F19', borderRadius: '8px', padding: '16px', border: '1px solid #1E293B' }}>
              <h4 style={{ fontSize: '12px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', marginBottom: '12px' }}>
                Land-Cover Transition Breakdown
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
                {Object.entries(result.specific_metrics.change_statistics).map(([cat, pct]: any) => (
                  <div key={cat} style={{ background: '#131B2E', padding: '10px 14px', borderRadius: '6px', border: '1px solid #1E293B' }}>
                    <div style={{ fontSize: '11px', color: '#94A3B8' }}>{cat}</div>
                    <div style={{ fontSize: '16px', fontWeight: 700, color: cat.includes('Built-up') ? '#F97316' : cat.includes('Loss') ? '#F43F5E' : '#10B981', marginTop: '2px' }}>
                      {pct}%
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
