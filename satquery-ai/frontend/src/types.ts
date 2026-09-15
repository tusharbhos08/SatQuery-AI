export interface ImageMetadata {
  width: number;
  height: number;
  bands: number;
  dtype: string;
  crs: string;
  bounds?: { west: number; east: number; south: number; north: number };
  resolution?: number;
  geotransform?: number[];
  acquisition_date?: string;
  nodata?: number;
  is_geotiff: boolean;
  file_size: number;
  warnings: string[];
}

export interface UploadedImage {
  id: string;
  filename: string;
  filepath: string;
  extension: string;
  file_size: number;
  modality: 'optical' | 'sar' | 'multispectral';
  is_geotiff: boolean;
  crs: string;
  resolution?: number;
  dimensions: { width: number; height: number };
  bands: number;
  preview_url: string;
  metadata: ImageMetadata;
  uploaded_at: string;
}

export interface TraceStep {
  step: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  time_ms: number;
  details: string;
}

export interface ExecutionTrace {
  steps: TraceStep[];
  total_time_ms: number;
}

export interface ConfidenceData {
  overall: number;
  overall_percentage: string;
  breakdown: {
    input_quality_confidence: number;
    model_inference_confidence: number;
    spatial_detection_confidence: number;
    modality_crosscheck_confidence: number;
  };
  warning?: string | null;
}

export interface VisualEvidence {
  type: string;
  primary_url?: string;
  primary_base64?: string;
  mask_url?: string;
  mask_base64?: string;
  optical_result_url?: string;
  optical_result_base64?: string;
  sar_result_url?: string;
  sar_result_base64?: string;
  before_url?: string;
  before_base64?: string;
  after_url?: string;
  after_base64?: string;
  side_by_side_url?: string;
  side_by_side_base64?: string;
  color_map_url?: string;
  color_map_base64?: string;
  [key: string]: any;
}

export interface AnalysisResult {
  id: string;
  query: string;
  task_type: string;
  answer: string;
  confidence: number;
  confidence_data: ConfidenceData;
  models_used: string[];
  execution_trace: ExecutionTrace;
  visual_evidence: VisualEvidence;
  specific_metrics: Record<string, any>;
  processing_time_ms: number;
  created_at: string;
}

export interface DemoScenario {
  id: number;
  title: string;
  description: string;
  image_ids: string[];
  query: string;
  expected_task: string;
}

export interface ModelInfo {
  name: string;
  version: string;
  task: string;
  input_type: string;
  output_type: string;
  supported_modalities: string[];
  confidence_support: boolean;
  device: string;
  status: string;
  description: string;
}
