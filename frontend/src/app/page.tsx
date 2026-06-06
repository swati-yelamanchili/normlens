'use client'

import { useState, useEffect, useCallback } from 'react'
import { Upload, FileText, BarChart3, Search, AlertTriangle, TrendingUp, ChevronRight, Loader2, Trash2, Shield, Database, Lock, Scale, Cpu, ChevronDown, ChevronUp, Lightbulb, XCircle } from 'lucide-react'
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
      const uploaded = await api.uploadContract(file)
      await loadContracts()
      if (uploaded?.id) {
        setSelectedContract(uploaded.id)
        handleAnalyze(uploaded.id)
      }
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
      if (status.status === 'analyzed') {
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
      pending:   'bg-slate-700 text-slate-300',
      analyzed:  'bg-emerald-900/60 text-emerald-400',
      failed:    'bg-red-900/60 text-red-400',
      parsing:   'bg-blue-900/60 text-blue-400',
      analyzing: 'bg-yellow-900/60 text-yellow-400',
      segmenting:'bg-indigo-900/60 text-indigo-400',
      classifying:'bg-purple-900/60 text-purple-400',
    }
    return colors[status] || 'bg-slate-700 text-slate-300'
  }

  const getSeverityColor = (s: string) => {
    const colors: Record<string, string> = {
      Critical: 'bg-red-950/70 text-red-300 border-red-800',
      High:     'bg-orange-950/70 text-orange-300 border-orange-800',
      Medium:   'bg-yellow-950/70 text-yellow-300 border-yellow-800',
      Low:      'bg-emerald-950/70 text-emerald-300 border-emerald-800',
    }
    return colors[s] || 'bg-slate-800 text-slate-300 border-slate-700'
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-950">
      <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-900/50">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold text-white">NormLens</h1>
              <span className="text-xs text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full border border-slate-700">Beta</span>
            </div>
            <label className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 cursor-pointer transition-colors text-sm font-medium shadow-lg shadow-indigo-900/40">
              <Upload className="w-4 h-4" />
              Upload Contract
              <input type="file" accept=".pdf,.docx" onChange={handleUpload} className="hidden" disabled={uploading} />
            </label>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 p-4 bg-red-950/50 border border-red-800 rounded-lg text-red-400 text-sm">{error}</div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-1">
            <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
              <h2 className="font-semibold text-slate-100 mb-3 flex items-center gap-2">
                <FileText className="w-4 h-4 text-indigo-400" />
                Contracts ({contracts.length})
              </h2>

              {uploading && (
                <div className="flex items-center gap-2 p-3 bg-blue-900/40 border border-blue-800 rounded-lg mb-3 text-sm text-blue-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </div>
              )}

              {contracts.length === 0 ? (
                <div className="text-center py-12 text-slate-500">
                  <FileText className="w-12 h-12 mx-auto mb-3 opacity-30" />
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
                          ? 'border-indigo-600 bg-indigo-900/30'
                          : 'border-slate-700 hover:border-slate-600 bg-slate-800/50'
                      }`}
                      onClick={() => {
                        setSelectedContract(c.id)
                        if (c.status === 'analyzed') {
                          api.getReport(c.id).then(setReport).catch((e) => setError(e.message))
                        }
                      }}
                    >
                      <div className="flex items-start justify-between">
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-slate-100 truncate">{c.filename}</p>
                          <div className="flex items-center gap-2 mt-1">
                            <span className="text-xs text-slate-500">{c.file_type?.toUpperCase()}</span>
                            <span className={`text-xs px-1.5 py-0.5 rounded-full ${getStatusColor(c.status)}`}>
                              {c.status}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 ml-2">
                          {c.status && !['analyzed', 'analyzing'].includes(c.status) ? (
                            <button
                              onClick={(e) => { e.stopPropagation(); handleAnalyze(c.id) }}
                              disabled={analyzing === c.id}
                              className="p-1.5 text-indigo-400 hover:bg-indigo-900/40 rounded"
                            >
                              {analyzing === c.id ? <Loader2 className="w-4 h-4 animate-spin" /> : <BarChart3 className="w-4 h-4" />}
                            </button>
                          ) : null}
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDelete(c.id) }}
                            className="p-1.5 text-slate-500 hover:text-red-400 hover:bg-red-900/30 rounded"
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

            <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 mt-4">
              <h2 className="font-semibold text-slate-100 mb-3 flex items-center gap-2">
                <Search className="w-4 h-4 text-indigo-400" />
                Search Clauses
              </h2>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="e.g., notice period, liability..."
                  className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <button
                  onClick={handleSearch}
                  className="px-3 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-500 text-sm"
                >
                  <Search className="w-4 h-4" />
                </button>
              </div>
              {searchResults.length > 0 && (
                <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
                  {searchResults.map((r, i) => (
                    <div key={i} className="p-2 bg-slate-800 rounded text-sm border border-slate-700">
                      <div className="flex items-center gap-2 mb-1">
                        {r.clause_type && (
                          <span className="text-xs bg-indigo-900/60 text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-800">
                            {r.clause_type}
                          </span>
                        )}
                        <span className="text-xs text-slate-500">Page {r.page_number}</span>
                        <span className="text-xs text-slate-500 ml-auto">{(r.relevance_score * 100).toFixed(0)}%</span>
                      </div>
                      <p className="text-slate-400 text-xs line-clamp-2">{r.clause_text}</p>
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
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-8 text-center text-slate-500">
                <BarChart3 className="w-16 h-16 mx-auto mb-4 opacity-20" />
                <p className="text-lg font-medium text-slate-300">Analysis pending</p>
                <p className="text-sm mt-1">Select a contract and click the analyze button to generate a report</p>
              </div>
            ) : (
              <div className="bg-slate-900 rounded-xl border border-slate-800 p-12 text-center">
                <div className="w-20 h-20 mx-auto mb-6 bg-indigo-900/40 rounded-2xl flex items-center justify-center border border-indigo-800/50">
                  <BarChart3 className="w-10 h-10 text-indigo-400" />
                </div>
                <h2 className="text-2xl font-semibold text-slate-100 mb-2">Welcome to NormLens</h2>
                <p className="text-slate-400 max-w-md mx-auto mb-8">
                  Upload a contract to analyze clauses, detect risks, identify outliers, and generate explainable reports.
                </p>
                <div className="grid grid-cols-3 gap-4 max-w-lg mx-auto">
                  {[
                    { icon: FileText, label: 'Clause Classification', desc: '30 CUAD clause types' },
                    { icon: AlertTriangle, label: 'Risk Detection', desc: 'Rule-based risk engine' },
                    { icon: TrendingUp, label: 'Outlier Detection', desc: 'Market norm benchmarking' },
                  ].map(({ icon: Icon, label, desc }) => (
                    <div key={label} className="p-3 bg-slate-800 rounded-lg border border-slate-700">
                      <Icon className="w-6 h-6 text-indigo-400 mx-auto mb-1" />
                      <p className="text-xs font-medium text-slate-200">{label}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{desc}</p>
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
  const data = report.report_data || {}
  const risk = data.risk_summary || {}
  const summary = data.contract_summary || {}
  const [activeGroup, setActiveGroup] = useState<string | null>(null)

  const riskColorMap: Record<string, string> = {
    Low: 'text-emerald-400', Moderate: 'text-yellow-400',
    High: 'text-orange-400', Critical: 'text-red-400',
  }

  const groupIcons: Record<string, any> = {
    Termination: Scale, Liability: Shield, Indemnification: Shield,
    IP: Cpu, Data: Database, Security: Lock, Confidentiality: Lock,
    General: FileText, Payment: FileText,
  }

  const contractTypeColors: Record<string, string> = {
    SaaS: 'bg-blue-900/50 text-blue-300 border border-blue-800',
    NDA: 'bg-purple-900/50 text-purple-300 border border-purple-800',
    MSA: 'bg-indigo-900/50 text-indigo-300 border border-indigo-800',
    Consulting: 'bg-teal-900/50 text-teal-300 border border-teal-800',
    Employment: 'bg-pink-900/50 text-pink-300 border border-pink-800',
    Vendor: 'bg-orange-900/50 text-orange-300 border border-orange-800',
    'License Agreement': 'bg-cyan-900/50 text-cyan-300 border border-cyan-800',
    'Government Contract': 'bg-slate-800 text-slate-300 border border-slate-700',
    'Professional Services': 'bg-emerald-900/50 text-emerald-300 border border-emerald-800',
    'General Commercial Agreement': 'bg-slate-800 text-slate-400 border border-slate-700',
  }

  const FindingCard = ({ f }: { f: any }) => {
    const [showClauses, setShowClauses] = useState(false)
    const clauses: string[] = f.supporting_clauses?.length ? f.supporting_clauses : (f.supporting_clause ? [f.supporting_clause] : [])
    return (
      <div className={`p-4 rounded-lg border ${getSeverityColor(f.severity)}`}>
        <div className="flex items-start justify-between mb-2">
          <div>
            <span className="font-medium text-sm">{f.risk_name}</span>
            <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full bg-black/20">{f.severity} ({f.points} pts)</span>
            {f.finding_category === 'Missing Protection' && (
              <span className="ml-2 text-xs px-1.5 py-0.5 rounded-full bg-black/20 border border-current/30">Missing</span>
            )}
          </div>
        </div>
        <p className="text-sm mb-2 opacity-90">{f.explanation}</p>
        {f.extracted_value && (
          <p className="text-xs font-mono bg-black/20 px-2 py-1 rounded mb-2">
            <span className="font-semibold">Detected: </span>{f.extracted_value}
          </p>
        )}
        {f.negotiation_recommendation && (
          <div className="mt-2 flex items-start gap-1.5 text-xs bg-black/20 rounded p-2 border border-current/20">
            <Lightbulb className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            <span>{f.negotiation_recommendation}</span>
          </div>
        )}
        {clauses.length > 0 && clauses[0] !== 'No matching clause found in contract' && (
          <details className="mt-2">
            <summary className="text-xs cursor-pointer hover:opacity-80 list-none flex items-center gap-1">
              <ChevronDown className="w-3 h-3" />
              View {clauses.length > 1 ? `${clauses.length} supporting clauses` : 'clause text'}
            </summary>
            <div className="mt-1 space-y-1">
              {clauses.map((c, i) => (
                <p key={i} className="text-xs p-2 bg-black/20 rounded border-l-2 border-current/40 opacity-80">{c}</p>
              ))}
            </div>
          </details>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-slate-100">Risk Analysis Report</h2>
            <p className="text-sm text-slate-400 mt-1">{summary.filename}</p>
            <div className="flex items-center gap-2 mt-2">
              {summary.contract_type && (
                <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                  contractTypeColors[summary.contract_type] || 'bg-slate-800 text-slate-400 border border-slate-700'
                }`}>
                  {summary.contract_type}
                  {summary.contract_type_confidence > 0 && ` · ${summary.contract_type_confidence}%`}
                </span>
              )}
            </div>
          </div>
          <div className={`text-right ${riskColorMap[risk.risk_level] || 'text-slate-400'}`}>
            <div className="text-4xl font-bold">{risk.total_score}</div>
            <div className="text-sm font-medium">{risk.risk_level} Risk</div>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-3 mb-4">
          {[
            { label: 'Clauses', value: summary.clause_count },
            { label: 'Findings', value: risk.finding_count },
            { label: 'Outliers', value: risk.outlier_count },
            { label: 'Pages', value: summary.page_count },
          ].map(({ label, value }) => (
            <div key={label} className="p-3 bg-slate-800 rounded-lg text-center border border-slate-700">
              <div className="text-2xl font-bold text-slate-100">{value}</div>
              <div className="text-xs text-slate-500">{label}</div>
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
        {risk.category_breakdown && (
          <div className="flex gap-2 mt-2">
            {Object.entries(risk.category_breakdown).map(([cat, count]) => (
              <div key={cat} className={`flex-1 p-2 rounded-lg text-center text-xs border ${
                cat === 'Missing Protection' ? 'bg-orange-950/50 border-orange-800 text-orange-400' : 'bg-red-950/50 border-red-800 text-red-400'
              }`}>
                <span className="font-semibold">{count as number}</span> {cat}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Missing Protections */}
      {data.missing_protections && data.missing_protections.length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-orange-900 p-6">
          <h3 className="font-semibold text-orange-400 mb-4 flex items-center gap-2">
            <XCircle className="w-4 h-4" />
            Missing Protections ({data.missing_protections.length})
          </h3>
          <div className="space-y-3">
            {data.missing_protections.map((f: any, i: number) => <FindingCard key={i} f={f} />)}
          </div>
        </div>
      )}

      {/* Risky Provisions — grouped by clause group */}
      {data.findings_by_group && Object.keys(data.findings_by_group).length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h3 className="font-semibold text-slate-100 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-500" />
            Risky Provisions
          </h3>
          <div className="space-y-2">
            {Object.entries(data.findings_by_group).map(([group, findings]: [string, any]) => {
              const Icon = groupIcons[group] || FileText
              const isOpen = activeGroup === group
              const maxSev = findings.reduce((acc: string, f: any) => {
                const order: Record<string, number> = { Critical: 4, High: 3, Medium: 2, Low: 1 }
                return (order[f.severity] || 0) > (order[acc] || 0) ? f.severity : acc
              }, 'Low')
              return (
                <div key={group} className="border border-slate-700 rounded-lg overflow-hidden">
                  <button
                    onClick={() => setActiveGroup(isOpen ? null : group)}
                    className="w-full flex items-center justify-between p-4 hover:bg-slate-800 text-left transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <Icon className="w-4 h-4 text-slate-400" />
                      <span className="font-medium text-sm text-slate-100">{group}</span>
                      <span className="text-xs text-slate-500">{findings.length} finding{findings.length !== 1 ? 's' : ''}</span>
                      <span className={`text-xs px-1.5 py-0.5 rounded-full border ${getSeverityColor(maxSev)}`}>{maxSev}</span>
                    </div>
                    {isOpen ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
                  </button>
                  {isOpen && (
                    <div className="px-4 pb-4 space-y-3 bg-slate-800/30">
                      {findings.map((f: any, i: number) => <FindingCard key={i} f={f} />)}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Outlier Detections */}
      {data.outlier_detections && data.outlier_detections.length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h3 className="font-semibold text-slate-100 mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-purple-400" />
            Market Outliers ({data.outlier_detections.length})
          </h3>
          <div className="space-y-3">
            {data.outlier_detections.map((o: any, i: number) => (
              <div key={i} className="p-4 rounded-lg border border-purple-800 bg-purple-950/40">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm text-slate-200">{o.attribute}</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                      o.severity === 'High' ? 'bg-red-900/60 text-red-400' : 'bg-yellow-900/60 text-yellow-400'
                    }`}>{o.severity}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-xs text-slate-500">P{Math.round(o.percentile_rank || 50)}</span>
                    <span className="text-xs text-slate-600 ml-2">z: {o.z_score?.toFixed(1)}</span>
                  </div>
                </div>
                <p className="text-sm text-slate-300">{o.explanation}</p>
                <div className="flex gap-4 mt-2 text-xs text-slate-500">
                  <span>Contract: <strong className="text-slate-300">{o.contract_value}</strong></span>
                  <span>Market median: <strong className="text-slate-300">{o.market_median}</strong></span>
                  {o.market_p95 && <span>P95: <strong className="text-slate-300">{o.market_p95}</strong></span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clause Classifications */}
      {data.clause_classifications && data.clause_classifications.length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h3 className="font-semibold text-slate-100 mb-4 flex items-center gap-2">
            <FileText className="w-4 h-4 text-indigo-400" />
            Clause Classifications ({data.clause_classifications.length})
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {data.clause_classifications.map((c: any, i: number) => (
              <div key={i} className="flex items-center gap-3 p-2 bg-slate-800 rounded-lg text-sm border border-slate-700/50">
                <span className="text-xs text-slate-600 w-8">#{c.clause_index + 1}</span>
                {c.clause_type && (
                  <span className="text-xs bg-indigo-900/60 text-indigo-400 px-2 py-0.5 rounded border border-indigo-800">{c.clause_type}</span>
                )}
                {c.clause_title && <span className="text-xs text-slate-400">{c.clause_title}</span>}
                <span className="text-xs text-slate-600 ml-auto">Page {c.page_number}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Negotiation Recommendations */}
      {data.recommendations && data.recommendations.length > 0 && (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h3 className="font-semibold text-slate-100 mb-4 flex items-center gap-2">
            <Lightbulb className="w-4 h-4 text-yellow-400" />
            Negotiation Recommendations
          </h3>
          <ul className="space-y-2">
            {data.recommendations.map((r: string, i: number) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-300">
                <ChevronRight className="w-4 h-4 text-indigo-400 mt-0.5 flex-shrink-0" />
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
