const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `API error: ${res.status}`)
  }
  return res.json()
}

export async function uploadContract(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/api/contracts/upload`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
  return res.json()
}

export async function listContracts() {
  return fetchAPI<any[]>('/api/contracts/')
}

export async function getContract(id: string) {
  return fetchAPI<any>(`/api/contracts/${id}`)
}

export async function deleteContract(id: string) {
  return fetchAPI<any>(`/api/contracts/${id}`, { method: 'DELETE' })
}

export async function analyzeContract(id: string) {
  return fetchAPI<any>(`/api/analysis/analyze/${id}`, { method: 'POST' })
}

export async function getAnalysisStatus(id: string) {
  return fetchAPI<any>(`/api/analysis/status/${id}`)
}

export async function getReport(id: string) {
  return fetchAPI<any>(`/api/reports/${id}`)
}

export async function getReportSummary(id: string) {
  return fetchAPI<any>(`/api/reports/${id}/summary`)
}

export async function searchClauses(q: string, contractId?: string) {
  const params = new URLSearchParams({ q, ...(contractId ? { contract_id: contractId } : {}) })
  return fetchAPI<any>(`/api/search/?${params}`)
}
