import React, { useState } from 'react';
import { 
  Upload, 
  Search, 
  Sparkles, 
  ShieldCheck, 
  FileText, 
  Layers, 
  Eye, 
  Sliders, 
  CheckCircle2, 
  Clock, 
  AlertCircle,
  Download
} from 'lucide-react';
import { uploadImage, executeQuery, generateReport } from '../api';
import { AnalysisResult, UploadedImage } from '../types';

export const AnalyzeView: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.8);
  const [error, setError] = useState<string | null>(null);
  const [reportUrl, setReportUrl] = useState<string | null>(null);

  const sampleQueries = [
    "Describe the land-cover and major structural objects visible in this observation.",
    "Highlight the water bodies and calculate surface area.",
    "Highlight the building structures and settlement clusters.",
    "Compute vegetation canopy coverage and photosynthetic health.",
    "Classify land-cover categories across standard classes.",
    "Find areas where vegetation decreased near roads within 500 meters.",
  ];

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
    setLoading(true);

    try {
      const data = await uploadImage(selectedFile);
      setUploadedImage(data.image);
    } catch (err: any) {
      setError(err.message || 'Failed to upload image');
    } finally {
      setLoading(false);
    }
  };

  const handleRunQuery = async (queryToRun?: string) => {
    const q = queryToRun || query;
    if (!q.trim()) return;
    if (!uploadedImage) {
      setError('Please upload a satellite image or GeoTIFF first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setReportUrl(null);

    try {
      const res = await executeQuery(q, [uploadedImage.id]);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!result) return;
    try {
      const rep = await generateReport(result.id);
      setReportUrl(`http://localhost:8000${rep.report_url}`);
      window.open(`http://localhost:8000${rep.report_url}`, '_blank');
    } catch (err: any) {
      alert('Report generation failed: ' + err.message);
    }
  };

  return (
    <div style={{ padding: '28px 32px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
        
        {/* Left Column: Upload & Query */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Uploader Card */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#F8FAFC', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Upload size={16} color="#38BDF8" />
              <span>Input Remote-Sensing Image / GeoTIFF</span>
            </h3>

            <label style={{
              border: '2px dashed #334155',
              borderRadius: '10px',
              padding: '28px 20px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              backgroundColor: 'rgba(15, 23, 42, 0.6)',
              transition: 'all 0.2s ease',
              marginBottom: '16px'
            }}>
              <input 
                type="file" 
                accept=".tif,.tiff,.geotiff,.png,.jpg,.jpeg" 
                onChange={handleFileUpload} 
                style={{ display: 'none' }} 
              />
              <Upload size={28} color="#38BDF8" style={{ marginBottom: '8px' }} />
              <span style={{ fontSize: '13.5px', fontWeight: 600, color: '#F8FAFC' }}>
                {file ? file.name : 'Click to Upload GeoTIFF / TIFF / Image'}
              </span>
              <span style={{ fontSize: '11px', color: '#64748B', marginTop: '4px' }}>
                Supports GeoTIFF, Optical, Multispectral (Bands 1-12), and SAR
              </span>
            </label>

            {/* Metadata Preview */}
            {uploadedImage && (
              <div style={{ backgroundColor: '#0B0F19', borderRadius: '8px', padding: '14px', border: '1px solid #1E293B' }}>
                <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', marginBottom: '8px' }}>
                  Extracted Geospatial Metadata
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '12px' }}>
                  <div><span style={{ color: '#64748B' }}>Modality:</span> <strong style={{ color: '#38BDF8' }}>{uploadedImage.modality.toUpperCase()}</strong></div>
                  <div><span style={{ color: '#64748B' }}>CRS:</span> <strong style={{ color: '#F8FAFC' }}>{uploadedImage.crs}</strong></div>
                  <div><span style={{ color: '#64748B' }}>Dimensions:</span> <strong style={{ color: '#F8FAFC' }}>{uploadedImage.dimensions.width} x {uploadedImage.dimensions.height}</strong></div>
                  <div><span style={{ color: '#64748B' }}>Bands:</span> <strong style={{ color: '#F8FAFC' }}>{uploadedImage.bands} Band(s)</strong></div>
                  {uploadedImage.resolution && (
                    <div><span style={{ color: '#64748B' }}>Resolution:</span> <strong style={{ color: '#10B981' }}>{uploadedImage.resolution} m/px</strong></div>
                  )}
                  <div><span style={{ color: '#64748B' }}>GeoTIFF:</span> <strong style={{ color: uploadedImage.is_geotiff ? '#10B981' : '#F59E0B' }}>{uploadedImage.is_geotiff ? 'Verified' : 'Standard'}</strong></div>
                </div>
              </div>
            )}
          </div>

          {/* Query Card */}
          <div className="glass-panel" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#F8FAFC', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Search size={16} color="#38BDF8" />
              <span>Natural-Language Query</span>
            </h3>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleRunQuery()}
                placeholder="Ask any remote-sensing question..."
                style={{
                  flex: 1,
                  backgroundColor: '#0B0F19',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  padding: '12px 16px',
                  color: '#F8FAFC',
                  fontSize: '13.5px',
                  outline: 'none'
                }}
              />
              <button 
                onClick={() => handleRunQuery()} 
                disabled={loading || !uploadedImage}
                className="btn-primary"
                style={{ opacity: (loading || !uploadedImage) ? 0.6 : 1 }}
              >
                {loading ? <Clock size={16} className="animate-spin" /> : <Sparkles size={16} />}
                <span>Analyze</span>
              </button>
            </div>

            {/* Suggestion Chips */}
            <div>
              <span style={{ fontSize: '11px', fontWeight: 600, color: '#64748B', textTransform: 'uppercase', display: 'block', marginBottom: '8px' }}>
                Suggested Remote-Sensing Prompts
              </span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {sampleQueries.map((sq, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setQuery(sq);
                      handleRunQuery(sq);
                    }}
                    style={{
                      background: '#0B0F19',
                      border: '1px solid #1E293B',
                      borderRadius: '6px',
                      padding: '8px 12px',
                      color: '#94A3B8',
                      fontSize: '12px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    "{sq}"
                  </button>
                ))}
              </div>
            </div>
          </div>

          {error && (
            <div style={{ backgroundColor: 'rgba(244, 63, 94, 0.1)', border: '1px solid #F43F5E', borderRadius: '8px', padding: '14px', display: 'flex', gap: '10px', alignItems: 'center' }}>
              <AlertCircle size={18} color="#F43F5E" />
              <span style={{ fontSize: '13px', color: '#F43F5E' }}>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Visual Evidence & Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Result Card */}
          <div className="glass-panel" style={{ padding: '24px', minHeight: '400px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#F8FAFC', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Eye size={16} color="#38BDF8" />
                <span>AI Grounded Assessment</span>
              </h3>
              {result && (
                <button onClick={handleGenerateReport} className="btn-secondary" style={{ padding: '6px 12px', fontSize: '12px' }}>
                  <Download size={14} />
                  <span>Download Report</span>
                </button>
              )}
            </div>

            {loading && (
              <div style={{ height: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '12px' }}>
                <div style={{ width: '40px', height: '40px', border: '3px solid #1E293B', borderTopColor: '#38BDF8', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                <span style={{ fontSize: '13px', color: '#94A3B8' }}>Autonomous Agent Executing Pipeline...</span>
              </div>
            )}

            {!loading && !result && (
              <div style={{ height: '300px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#64748B', textAlign: 'center' }}>
                <Layers size={36} color="#334155" style={{ marginBottom: '12px' }} />
                <p style={{ fontSize: '13.5px' }}>Upload a satellite image and enter a query to run the agentic assistant.</p>
              </div>
            )}

            {result && (
              <div>
                {/* Answer Box */}
                <div style={{
                  backgroundColor: '#0F172A',
                  border: '1px solid #38BDF8',
                  borderRadius: '8px',
                  padding: '16px',
                  fontSize: '14px',
                  lineHeight: 1.6,
                  color: '#F8FAFC',
                  marginBottom: '18px'
                }}>
                  {result.answer}
                </div>

                {/* Hierarchical Confidence Bar */}
                <div style={{ backgroundColor: '#0B0F19', borderRadius: '8px', padding: '14px', border: '1px solid #1E293B', marginBottom: '18px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <ShieldCheck size={16} color="#10B981" />
                      <span style={{ fontSize: '12px', fontWeight: 600, color: '#F8FAFC' }}>Overall Confidence</span>
                    </div>
                    <span style={{ fontSize: '13px', fontWeight: 700, color: '#10B981' }}>
                      {result.confidence_data.overall_percentage}
                    </span>
                  </div>
                  <div style={{ width: '100%', height: '6px', backgroundColor: '#1E293B', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ width: `${result.confidence * 100}%`, height: '100%', backgroundColor: '#10B981', transition: 'width 0.5s ease' }}></div>
                  </div>
                </div>

                {/* Visual Evidence Viewer */}
                {result.visual_evidence?.primary_base64 && (
                  <div style={{ marginBottom: '18px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontSize: '12px', fontWeight: 600, color: '#94A3B8' }}>Visual Raster Evidence</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Sliders size={13} color="#64748B" />
                        <span style={{ fontSize: '11px', color: '#64748B' }}>Opacity:</span>
                        <input
                          type="range"
                          min="0.2"
                          max="1.0"
                          step="0.05"
                          value={overlayOpacity}
                          onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                          style={{ width: '80px', accentColor: '#38BDF8' }}
                        />
                      </div>
                    </div>
                    <div style={{ position: 'relative', borderRadius: '8px', overflow: 'hidden', border: '1px solid #334155' }}>
                      <img
                        src={result.visual_evidence.primary_base64}
                        alt="Visual Evidence"
                        style={{ width: '100%', height: 'auto', display: 'block', opacity: overlayOpacity }}
                      />
                    </div>
                  </div>
                )}

                {/* Observable Execution Trace Panel */}
                <div style={{ backgroundColor: '#0B0F19', borderRadius: '8px', padding: '14px', border: '1px solid #1E293B' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', marginBottom: '10px', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Observable Execution Trace</span>
                    <span>Total: {result.execution_trace.total_time_ms} ms</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {result.execution_trace.steps.map((step, idx) => (
                      <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                        <CheckCircle2 size={13} color="#10B981" />
                        <strong style={{ color: '#E2E8F0' }}>{step.step}</strong>
                        {step.details && <span style={{ color: '#64748B' }}>— {step.details}</span>}
                        {step.time_ms > 0 && <span style={{ color: '#38BDF8', marginLeft: 'auto', fontSize: '11px' }}>{step.time_ms}ms</span>}
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};
