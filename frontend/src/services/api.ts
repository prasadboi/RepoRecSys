import axios from 'axios'

export interface Recommendation {
  project_id: number | string
  score: number
  github_url?: string
  language?: string
  watchers?: number
  is_fallback?: boolean
}

export interface RecommendationResponse {
  user_id: string
  recommendations: Recommendation[]
}

export async function getRecommendations(
  userId: string,
  topK: number,
  apiUrl: string,
  userName: string = ''
): Promise<RecommendationResponse> {
  const url = `${apiUrl.replace(/\/$/, '')}/recommend`
  const resp = await axios.post<RecommendationResponse>(
    url,
    { user_id: userId, user_name: userName, top_k: topK },
    { timeout: 30000 },
  )
  return resp.data
}

export async function triggerIngest(apiUrl: string): Promise<{ status: string }> {
  const url = `${apiUrl.replace(/\/$/, '')}/system/ingest`
  const resp = await axios.post(url, {}, { timeout: 30000 })
  return resp.data
}

export async function triggerTrain(apiUrl: string): Promise<{ status: string }> {
  const url = `${apiUrl.replace(/\/$/, '')}/system/train`
  const resp = await axios.post(url, {}, { timeout: 30000 })
  return resp.data
}

export async function triggerReload(apiUrl: string): Promise<{ status: string }> {
  const url = `${apiUrl.replace(/\/$/, '')}/system/reload`
  const resp = await axios.post(url, {}, { timeout: 30000 })
  return resp.data
}

export async function fetchStatus(apiUrl: string): Promise<{ last_action: string; last_update: string | null; logs: string[] }> {
  const url = `${apiUrl.replace(/\/$/, '')}/system/status`
  const resp = await axios.get(url, { timeout: 10000 })
  return resp.data
}

export async function checkHealth(apiUrl: string): Promise<boolean> {
  try {
    const url = `${apiUrl.replace(/\/$/, '')}/health`
    const resp = await axios.get(url, { timeout: 5000 })
    return resp.status === 200
  } catch {
    return false
  }
}

