'use client'

import { useState, useEffect, useCallback } from 'react'
import { Upload, FileText, BarChart3, Search, AlertTriangle, TrendingUp, ArrowRight, ChevronRight, Loader2, Trash2, ExternalLink } from 'lucide-react'
import * as api from '@/lib/api'
import type { Contract } from '@/types'

export default function Home() {
  const [contracts, setContracts] = useState<Contract[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState<string | null>(null)
  const [selectedContract, setSelectedContract] = useState<string | null>(null)
  const [report, setReport] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])

  const loadContracts = useCallback(async () => {
    try {
      const data = await api.listContracts()
      setContracts(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadContracts() }, [loadContracts])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      await api.uploadContract(file)
      await loadContracts()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const handleAnalyze = async (id: string) => {
    setAnalyzing(id)
    setError(null)
    try {
      await api.analyzeContract(id)
      const status = await api.getAnalysisStatus(id)
      if (status.status === 'analyzed' || status.status === 'completed') {
        const r = await api.getReport(id)
        setReport(r)
        setSelectedContract(id)
      }
    } catch (e: any) {
      setError(e.message)
    } finally {
      setAnalyzing(null)
    }
  }

  const handleDelete = async (id: string) => {
    try {
      await api.deleteContract(id)
      await loadContracts()
      if (selectedContract === id) {
        setSelectedContract(null)
        setReport(null)
      }
    } catch (e: any) {
      setError(e.message)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    try {
      const res = await api.searchClauses(searchQuery, selectedContract || undefined)
      setSearchResults(res.results || [])
    } catch (e: any) {
      setError(e.message)
    }
  }

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      pending: 'bg-gray-100 text-gray-600',
      analyzed: 'bg-green-100 text-green-700',
      failed: 'bg-red-100 text-red-700',
      parsing: 'bg-blue-100 text-blue-700',
      analyzing: 'bg-yellow-100 text-yellow-700',
    }
    return colors[status] || 'bg-gray-100 text-gray-600'
  }

  const getSeverityColor = (s: string) => {
    const colors: Record<string, string> = {
      Critical: 'bg-red-100 text-red-700 border-red-200',
      High: 'bg-orange-100 text-orange-700 border-orange-200',
      Medium: 'bg-yellow-100 text-yellow-700 border-yellow-200',
      Low: 'bg-green-100 text-green-700 border-green-200',
    }
    return colors[s] || 'bg-gray-100 text-gray-600'
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold text-gray-900">NormLens</h1>
              <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">Beta</span>
            </div>
            <label className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 cursor-pointer transition-colors text-sm font-medium">
              <Upload className="w-4 h-4" />
              Upload Contract
              <input type="file" accept=".pdf,.docx" onChange={handleUpload} className="hidden" disabled={uploading} />
            </label>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">{error}</div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <h2 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary-600" />
                Contracts ({contracts.length})
              </h2>

              {uploading && (
                <div className="flex items-center gap-2 p-3 bg-blue-50 rounded-lg mb-3 text-sm text-blue-700">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </div>
              )}

              {contracts.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">No contracts uploaded yet</p>
                  <p className="text-xs mt-1">Upload a PDF or DOCX to get started</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {contracts.map((c) => (
                    <div
                      key={c.id}
                      className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedContract === c.id
                          ? 'border-primary-300 bg-primary-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => {
                        setSelectedContract(c.id)
                        if (c.status === 'analyzed') {
                          api.getReport(c.id).then(setReport).catch(() => {})
                        }
                      }}
                    >
                      <div className="flex items-start justify-between">
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-gray-900 truncate">{c.filename}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-gray-400">{c.file_type?.toUpperCase()}</span>
                            <span className={`text-xs px-1.5 py-0.5 rounded-full ${getStatusColor(c.status)}`}>
                              {c.status}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 ml-2">
                          {c.status === 'parsed' || c.status === 'segmented' || c.status === 'classified' || (c.status && !['analyzed', 'pending', 'failed', 'analyzing'].includes(c.status)) ? (
                            <button
                              onClick={(e) => { e.stopPropagation(); handleAnalyze(c.id) }}
                              disabled={analyzing === c.id}
                              className="p-1.5 text-primary-600 hover:bg-primary-50 rounded"
                            >
                              {analyzing === c.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <BarChart3 className="w-4 h-4" />}
                            </button>
                          ) : null}
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDelete(c.id) }}
                            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-white rounded-xl border border-gray-200 p-4 mt-4">
              <h2 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Search className="w-4 h-4 text-primary-600" />
                Search Clauses
              </h2>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="e.g., notice period, liability..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <button
                  onClick={handleSearch}
                  className="px-3 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm"
                >
                  <Search className="w-4 h-4" />
                </button>
              </div>
              {searchResults.length > 0 && (
                <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
                  {searchResults.map((r, i) => (
                    <div key={i} className="p-2 bg-gray-50 rounded text-sm">
                      <div className="flex items-center gap-2 mb-1">
                        {r.clause_type && (
                          <span className="text-xs bg-primary-100 text-primary-700 px-1.5 py-0.5 rounded">
                            {r.clause_type}
                          </span>
                        )}
                        <span className="text-xs text-gray-400">Page {r.page_number}</span>
                        <span className="text-xs text-gray-400 ml-auto">{(r.relevance_score * 100).toFixed(0)}%</span>
                      </div>
                      <p className="text-gray-600 text-xs line-clamp-2">{r.clause_text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2">
            {report ? (
              <ReportView report={report} getSeverityColor={getSeverityColor} />
            ) : selectedContract ? (
              <div className="bg-white rounded-xl border border-gray-200 p-8 text-center text-gray-400">
                <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-30" />
                <p className="text-lg font-medium">Analysis pending</p>
                <p className="text-sm mt-1">Select a contract and click the analyze button to generate a report</p>
              </div>
            ) : (
              <div className="bg-white rounded-xl border border-gray-200 p-12 text-center text-gray-400">
                <div className="w-20 h-20 mx-auto mb-6 bg-primary-50 rounded-2xl flex items-center justify-center">
                  <BarChart3 className="w-10 h-10 text-primary-400" />
                </div>
                <h2 className="text-2xl font-semibold text-gray-900 mb-2">Welcome to NormLens</h2>
                <p className="text-gray-500 max-w-md mx-auto mb-8">
                  Upload a contract to analyze clauses, detect risks, identify outliers, and generate explainable reports.
                </p>
                <div className="grid grid-cols-3 gap-4 max-w-lg mx-auto">
                  {[
                    { icon: FileText, label: 'Clause Classification', desc: '30 CUAD clause types' },
                    { icon: AlertTriangle, label: 'Risk Detection', desc: 'Rule-based risk engine' },
                    { icon: TrendingUp, label: 'Outlier Detection', desc: 'Market norm benchmarking' },
                  ].map(({ icon: Icon, label, desc }) => (
                    <div key={label} className="p-3 bg-gray-50 rounded-lg">
                      <Icon className="w-6 h-6 text-primary-600 mx-auto mb-1" />
                      <p className="text-xs font-medium text-gray-900">{label}</p>
                      <p className="text-xs text-gray-400 mt-0.5">{desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

function ReportView({ report, getSeverityColor }: { report: any; getSeverityColor: (s: string) => string }) {
  const data = report.report_data
  const risk = data.risk_summary

  const riskColorMap: Record<string, string> = {
    Low: 'text-risk-low',
    Moderate: 'text-risk-moderate',
    High: 'text-risk-high',
    Critical: 'text-risk-critical',
  }

  const riskBgMap: Record<string, string> = {
    Low: 'bg-risk-low',
    Moderate: 'bg-risk-moderate',
    High: 'bg-risk-high',
    Critical: 'bg-risk-critical',
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Risk Analysis Report</h2>
            <p className="text-sm text-gray-500 mt-1">{data.contract_summary.filename}</p>
          </div>
          <div className={`text-right ${riskColorMap[risk.risk_level] || 'text-gray-600'}`}>
            <div className="text-4xl font-bold">{risk.total_score}</div>
            <div className="text-sm font-medium">{risk.risk_level}</div>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Clauses', value: data.contract_summary.clause_count },
            { label: 'Findings', value: risk.finding_count },
            { label: 'Outliers', value: risk.outlier_count },
            { label: 'Pages', value: data.contract_summary.page_count },
          ].map(({ label, value }) => (
            <div key={label} className="p-3 bg-gray-50 rounded-lg text-center">
              <div className="text-2xl font-bold text-gray-900">{value}</div>
              <div className="text-xs text-gray-500">{label}</div>
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          {Object.entries(risk.severity_breakdown || {}).map(([sev, count]) => (
            <div key={sev} className={`flex-1 p-3 rounded-lg border ${getSeverityColor(sev)}`}>
              <div className="text-lg font-bold">{count as number}</div>
              <div className="text-xs">{sev}</div>
            </div>
          ))}
        </div>
      </div>

      {data.clause_classifications && data.clause_classifications.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary-600" />
            Clause Classifications ({data.clause_classifications.length})
          </h3>
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {data.clause_classifications.map((c: any, i: number) => (
              <div key={i} className="flex items-center gap-3 p-2 bg-gray-50 rounded-lg text-sm">
                <span className="text-xs text-gray-400 w-8">#{c.clause_index + 1}</span>
                {c.clause_type && (
                  <span className="text-xs bg-primary-100 text-primary-700 px-2 py-0.5 rounded">{c.clause_type}</span>
                )}
                {c.clause_title && <span className="text-xs text-gray-600">{c.clause_title}</span>}
                <span className="text-xs text-gray-400 ml-auto">Page {c.page_number}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {risk.finding_count > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-risk-high" />
            Risk Findings ({risk.finding_count})
          </h3>
          <div className="space-y-3">
            {data.risk_findings.map((f: any, i: number) => (
              <div key={i} className={`p-4 rounded-lg border ${getSeverityColor(f.severity)}`}>
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <span className="font-medium text-sm">{f.risk_name}</span>
                    <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full bg-white/50">{f.severity} ({f.points} pts)</span>
                  </div>
                </div>
                <p className="text-sm mb-1">{f.explanation}</p>
                {f.extracted_value && (
                  <p className="text-xs mt-1">
                    <span className="font-medium">Extracted:</span> {f.extracted_value}
                  </p>
                )}
                {f.supporting_clause && (
                  <details className="mt-2">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">View clause text</summary>
                    <p className="text-xs text-gray-500 mt-1 p-2 bg-white/50 rounded">{f.supporting_clause}</p>
                  </details>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {data.outlier_detections && data.outlier_detections.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-risk-high" />
            Outlier Detections ({data.outlier_detections.length})
          </h3>
          <div className="space-y-3">
            {data.outlier_detections.map((o: any, i: number) => (
              <div key={i} className="p-4 rounded-lg border border-purple-200 bg-purple-50">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{o.attribute}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                      o.severity === 'High' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'
                    }`}>
                      {o.severity}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className="text-xs text-gray-500">P{Math.round(o.percentile_rank || 50)}</span>
                    <span className="text-xs text-gray-400 ml-2">z: {o.z_score?.toFixed(1)}</span>
                  </div>
                </div>
                <p className="text-sm text-gray-700">{o.explanation}</p>
                <div className="flex gap-4 mt-2 text-xs text-gray-500">
                  <span>Contract: <strong>{o.contract_value}</strong></span>
                  <span>Market median: <strong>{o.market_median}</strong></span>
                  {o.market_p95 && <span>P95: <strong>{o.market_p95}</strong></span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.recommendations && data.recommendations.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-4">Recommendations</h3>
          <ul className="space-y-2">
            {data.recommendations.map((r: string, i: number) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <ChevronRight className="w-4 h-4 text-primary-600 mt-0.5 flex-shrink-0" />
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Methodology</h3>
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(data.methodology || {}).map(([key, val]) => (
            <div key={key} className="p-3 bg-gray-50 rounded-lg text-sm">
              <span className="font-medium text-gray-900 block">{key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}</span>
              <span className="text-gray-500 text-xs">{val as string}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
