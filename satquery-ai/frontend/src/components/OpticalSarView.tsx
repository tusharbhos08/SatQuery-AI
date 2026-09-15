import React, { useState } from 'react';
import { Layers, Upload, Sparkles, Zap, ShieldCheck } from 'lucide-react';
import { uploadImage, executeQuery } from '../api';
import { AnalysisResult, UploadedImage } from '../types';

export const OpticalSarView: React.FC = () => {
  const [optImg, setOptImg] = useState<UploadedImage | null>(null);
  const [sarImg, setSarImg] = useState<UploadedImage | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleUpload = async (file: File, type: 'optical' | 'sar') => {
    try {
      const data = await uploadImage(file);
      if (type === 'optical') setOptImg(data.image);
      else setSarImg(data.image);
    } catch (e: any) {
      alert('Upload failed: ' + e.message);
    }
  };

  const handleRunFusion = async () => {
    if (!optImg || !sarImg) {
      alert('Please upload both Optical and SAR GeoTIFF images');
      return;
    }

    setLoading(true);
    try {
      const res = await executeQuery("Use the optical and SAR images together to identify built-up and water-covered regions.", [optImg.id, sarImg.id]);
      setResult(res);
    } catch (e: any) {
      alert('Fusion failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '28px 32px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Upload Pair Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '24px' }}>
        
        {/* Optical Upload */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#38BDF8', marginBottom: '12px' }}>
            1. Optical / Multispectral GeoTIFF (Cartosat / Sentinel-2)
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
            <input type="file" onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0], 'optical')} style={{ display: 'none' }} />
            <Upload size={24} color="#38BDF8" style={{ marginBottom: '6px' }} />
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>
              {optImg ? optImg.filename : 'Upload Optical GeoTIFF'}
            </span>
          </label>
        </div>

        {/* SAR Upload */}
        <div className="glass-panel" style={{ padding: '20px' }}>
          <h4 style={{ fontSize: '14px', fontWeight: 700, color: '#A855F7', marginBottom: '12px' }}>
            2. SAR Radar GeoTIFF (RISAT / Sentinel-1)
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
            <input type="file" onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0], 'sar')} style={{ display: 'none' }} />
            <Upload size={24} color="#A855F7" style={{ marginBottom: '6px' }} />
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>
              {sarImg ? sarImg.filename : 'Upload SAR GeoTIFF'}
            </span>
          </label>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginBottom: '28px' }}>
        <button 
          onClick={handleRunFusion} 
          disabled={loading || !optImg || !sarImg}
          className="btn-primary"
          style={{ padding: '12px 28px', fontSize: '14.5px', opacity: (!optImg || !sarImg) ? 0.6 : 1 }}
        >
          <Zap size={18} />
          <span>Run Multimodal Feature-Level Fusion Analysis</span>
        </button>
      </div>

      {result && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#F8FAFC' }}>Multimodal Fusion Results</h3>
            <span style={{ fontSize: '12.5px', fontWeight: 700, color: '#A855F7', backgroundColor: 'rgba(168, 85, 247, 0.15)', padding: '4px 10px', borderRadius: '6px' }}>
              Cross-Modal Agreement: {result.specific_metrics?.modality_agreement}%
            </span>
          </div>

          <div style={{ backgroundColor: '#0F172A', border: '1px solid #A855F7', borderRadius: '8px', padding: '16px', fontSize: '14px', lineHeight: 1.6, marginBottom: '20px' }}>
            {result.answer}
          </div>

          {/* 3-Way Triptych Display */}
          <div style={{ marginBottom: '24px' }}>
            <h4 style={{ fontSize: '13px', fontWeight: 600, color: '#94A3B8', marginBottom: '12px' }}>
              Multimodal Inspection: Optical Feature Map • SAR Radar Classification • Fused Consensus Map
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
              {result.visual_evidence?.optical_result_base64 && (
                <div style={{ background: '#0B0F19', padding: '10px', borderRadius: '8px', border: '1px solid #1E293B', textAlign: 'center' }}>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#38BDF8', display: 'block', marginBottom: '6px' }}>OPTICAL SPECTRAL CLASSIFICATION</span>
                  <img src={result.visual_evidence.optical_result_base64} style={{ width: '100%', borderRadius: '6px' }} />
                </div>
              )}
              {result.visual_evidence?.sar_result_base64 && (
                <div style={{ background: '#0B0F19', padding: '10px', borderRadius: '8px', border: '1px solid #1E293B', textAlign: 'center' }}>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#A855F7', display: 'block', marginBottom: '6px' }}>SAR RADAR BACKSCATTER</span>
                  <img src={result.visual_evidence.sar_result_base64} style={{ width: '100%', borderRadius: '6px' }} />
                </div>
              )}
              {result.visual_evidence?.primary_base64 && (
                <div style={{ background: '#0B0F19', padding: '10px', borderRadius: '8px', border: '1px solid #38BDF8', textAlign: 'center' }}>
                  <span style={{ fontSize: '11px', fontWeight: 700, color: '#10B981', display: 'block', marginBottom: '6px' }}>FUSED CONSENSUS MAP</span>
                  <img src={result.visual_evidence.primary_base64} style={{ width: '100%', borderRadius: '6px' }} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
