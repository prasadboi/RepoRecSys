import axios from 'axios'
import { getMockRecommendations, isMockModeEnabled } from './mockData'

export interface Recommendation {
  project_id: number
  github_url: string
  score: number
  language: string
  watchers: number
}

export interface RecommendationResponse {
  user_id: number
  recommendations: Recommendation[]
}

export interface RecommendationRequest {
  user_id: number
  top_k: number
}

/**
 * Get recommendations for a user from the FastAPI backend
 * Falls back to mock data if mock mode is enabled or API is unavailable
 */
export async function getRecommendations(
  userId: number,
  topK: number,
  apiUrl: string = 'http://localhost:8000',
  useMock: boolean = false
): Promise<RecommendationResponse> {
  // Use mock data if explicitly requested or mock mode is enabled
  if (useMock || isMockModeEnabled()) {
    console.log('Using mock data for recommendations')
    return getMockRecommendations(userId, topK)
  }

  try {
    const response = await axios.post<RecommendationResponse>(
      `${apiUrl}/recommend`,
      {
        user_id: userId,
        top_k: topK,
      } as RecommendationRequest,
      {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 30000, // 30 seconds timeout
      }
    )
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.response) {
        // Server responded with error status
        const status = error.response.status
        const message = error.response.data?.detail || error.message
        throw new Error(`API Error (${status}): ${message}`)
      } else if (error.request) {
        // Request made but no response - fall back to mock data
        console.warn('API unavailable, falling back to mock data')
        return getMockRecommendations(userId, topK)
      } else {
        // Error setting up request
        throw new Error(`Request error: ${error.message}`)
      }
    }
    throw new Error('Unknown error occurred')
  }
}

/**
 * Health check endpoint
 */
export async function checkHealth(apiUrl: string = 'http://localhost:8000'): Promise<boolean> {
  try {
    const response = await axios.get(`${apiUrl}/health`, { timeout: 5000 })
    return response.data.status === 'healthy'
  } catch {
    return false
  }
}

