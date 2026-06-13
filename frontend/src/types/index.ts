export interface Contract {
  id: string
  filename: string
  file_type: string
  status: string
  page_count?: string
  created_at: string
}

export interface Clause {
  clause_index: number
  clause_title: string | null
  clause_text: string
  page_number: number
  clause_type: string | null
  classification_confidence: number | null
  attributes: Record<string, any> | null
}

export interface RiskFinding {
  rule_id: string
  risk_name: string
  severity: string
  points: number
  supporting_clause: string
  extracted_value: string
  explanation: string
  clause_index: number | null
  clause_type: string | null
}

export interface OutlierDetection {
  type: string
  attribute: string
  severity: string
  contract_value: string
  market_median: number | null
  market_p95: number | null
  market_p5: number | null
  percentile_rank: number | null
  z_score: number | null
  clause_type: string
  explanation: string
}

export interface AnalysisReport {
  report_id: string
  contract_id: string
  total_risk_score: number
  risk_level: string
  clause_count: number
  risk_finding_count: number
  outlier_count: number
  report_data: {
    contract_summary: {
      filename: string
      file_type: string
      page_count: string
      clause_count: number
    }
    risk_summary: {
      total_score: number
      risk_level: string
      finding_count: number
      outlier_count: number
      severity_breakdown: Record<string, number>
    }
    clause_classifications: Array<{
      clause_index: number
      clause_title: string | null
      clause_type: string
      page_number: number
      text_preview: string
    }>
    risk_findings: Array<{
      risk_name: string
      severity: string
      points: number
      explanation: string
      supporting_clause: string
      extracted_value: string
      clause_index: number | null
    }>
    outlier_detections: OutlierDetection[]
    recommendations: string[]
    methodology: Record<string, string>
  }
  generated_at: string
}

export interface SearchResult {
  clause_index: number
  clause_title: string | null
  clause_type: string | null
  clause_text: string
  page_number: number
  relevance_score: number
}

export interface AnalysisStatus {
  contract_id: string
  status: string
  error_message: string | null
}
