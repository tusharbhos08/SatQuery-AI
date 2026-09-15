import React, { useState } from 'react';
import { Droplet, Upload, Sparkles, Download, Layers } from 'lucide-react';
import { uploadImage, executeQuery } from '../api';
import { AnalysisResult, UploadedImage } from '../types';

export const WaterFinderView: React.FC = () => {
  const [img, setImg] = useState<UploadedImage | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleUpload = async (file: File) => {
    try {
      const data = await uploadImage(file);
      setImg(data.image);
    } catch (e: any) {
      alert('Upload failed: ' + e.message);
    }
  };

  const handleDetectWater = async () => {
    if (!img) {
      alert('Please upload a satellite image first');
      return;
    }
    setLoading(true);
    try {
      const res = await executeQuery("Find and segment all water bodies and calculate estimated surface area in km².", [img.id]);
      setResult(res);
    } catch (e: any) {
      alert('Detection failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExportGeoJSON = () => {
    if (!result?.specific_metrics?.polygons) return;
    const geojson = {
      type: "FeatureCollection",
      features: result.specific_metrics.polygons.map((poly: any, idx: number) => ({
        type: "Feature",
        id: idx + 1,
        properties: {
          area_pixels: poly.area_pixels,
          type: "WaterBody"
        },
        geometry: {
          type: "Polygon",
          coordinates: [poly.coordinates]
        }
      }))
    };

    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `water_bodies_extracted_${Date.now()}.geojson`;
    a.click();
  };

  return (
    <div style={{ padding: '28px 32px', maxWidth: '1400px', margin: '0 auto' }}>
      <div className="glass-panel" style={{ padding: '24px', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#38BDF8', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Droplet size={18} color="#38BDF8" />
          <span>Dedicated Automated Water Body Finder</span>
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '16px', alignItems: 'center' }}>
          <label style={{
            border: '2px dashed #334155',
            borderRadius: '8px',
            padding: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            cursor: 'pointer',
            backgroundColor: 'rgba(15, 23, 42, 0.5)'
          }}>
            <input type="file" onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])} style={{ display: 'none' }} />
            <Upload size={20} color="#38BDF8" />
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#F8FAFC' }}>
              {img ? img.filename : 'Upload GeoTIFF / Coastal / River Basin Image'}
            </span>
          </label>

          <button 
            onClick={handleDetectWater} 
            disabled={loading || !img}
            className="btn-primary"
            style={{ padding: '14px 24px', opacity: (!img) ? 0.6 : 1 }}
          >
            <Droplet size={16} />
            <span>Detect Water Bodies</span>
          </button>
        </div>
      </div>

      {result && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          {/* Stats Bar */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '14px', marginBottom: '20px' }}>
            <div style={{ background: '#0B0F19', padding: '14px', borderRadius: '8px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '11px', color: '#64748B', textTransform: 'uppercase' }}>Water Bodies Detected</div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#38BDF8', marginTop: '4px' }}>
                {result.specific_metrics?.water_bodies_count || 0}
              </div>
            </div>
            <div style={{ background: '#0B0F19', padding: '14px', borderRadius: '8px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '11px', color: '#64748B', textTransform: 'uppercase' }}>Surface Coverage</div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#10B981', marginTop: '4px' }}>
                {result.specific_metrics?.coverage_percent || 0}%
              </div>
            </div>
            <div style={{ background: '#0B0F19', padding: '14px', borderRadius: '8px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '11px', color: '#64748B', textTransform: 'uppercase' }}>Est. Total Area</div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#F8FAFC', marginTop: '4px' }}>
                {result.specific_metrics?.total_area_km2 ? `${result.specific_metrics.total_area_km2} km²` : `${result.specific_metrics?.total_area_pixels} px`}
              </div>
            </div>
            <div style={{ background: '#0B0F19', padding: '14px', borderRadius: '8px', border: '1px solid #1E293B' }}>
              <div style={{ fontSize: '11px', color: '#64748B', textTransform: 'uppercase' }}>Model Confidence</div>
              <div style={{ fontSize: '22px', fontWeight: 800, color: '#38BDF8', marginTop: '4px' }}>
                {result.confidence_data.overall_percentage}
              </div>
            </div>
          </div>

          {/* Action Bar */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginBottom: '16px' }}>
            <button onClick={handleExportGeoJSON} className="btn-secondary" style={{ fontSize: '12px', padding: '8px 14px' }}>
              <Download size={14} />
              <span>Export GeoJSON Vector Polygons</span>
            </button>
          </div>

          {/* Mask Overlay */}
          {result.visual_evidence?.primary_base64 && (
            <div style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid #334155' }}>
              <img src={result.visual_evidence.primary_base64} alt="Water Body Overlay" style={{ width: '100%', height: 'auto', display: 'block' }} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
