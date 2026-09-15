import { AnalysisResult, DemoScenario, ModelInfo, UploadedImage } from './types';

const API_BASE = 'http://localhost:8000/api';

export async function uploadImage(file: File): Promise<{ image: UploadedImage; validation: any }> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/images/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Failed to upload image');
  }

  const data = await res.json();
  return data;
}

export async function executeQuery(query: string, imageIds: string[]): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, image_ids: imageIds }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Query execution failed');
  }

  return await res.json();
}

export async function getDemoScenarios(): Promise<DemoScenario[]> {
  const res = await fetch(`${API_BASE}/demo/scenarios`);
  if (!res.ok) throw new Error('Failed to fetch demo scenarios');
  const data = await res.json();
  return data.scenarios;
}

export async function getModels(): Promise<{ models: ModelInfo[]; adaptation_status: any }> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error('Failed to fetch model registry');
  return await res.json();
}

export async function getDashboardStats(): Promise<any> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error('Failed to fetch dashboard stats');
  return await res.json();
}

export async function generateReport(analysisId: string): Promise<{ report_url: string }> {
  const res = await fetch(`${API_BASE}/reports/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ analysis_id: analysisId }),
  });
  if (!res.ok) throw new Error('Failed to generate report');
  return await res.json();
}

export async function startTraining(config: any): Promise<any> {
  const res = await fetch(`${API_BASE}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error('Failed to start training');
  return await res.json();
}
