import { RecommendationResponse } from './api'

/**
 * Mock data for development and testing
 * This simulates the FastAPI backend response
 */

const MOCK_RECOMMENDATIONS = [
  {
    project_id: 28457823,
    github_url: 'https://github.com/tensorflow/tensorflow',
    score: 0.95,
    language: 'C++',
    watchers: 185000,
  },
  {
    project_id: 10270250,
    github_url: 'https://github.com/facebook/react',
    score: 0.92,
    language: 'JavaScript',
    watchers: 220000,
  },
  {
    project_id: 11730342,
    github_url: 'https://github.com/kubernetes/kubernetes',
    score: 0.89,
    language: 'Go',
    watchers: 110000,
  },
  {
    project_id: 2126244,
    github_url: 'https://github.com/torvalds/linux',
    score: 0.87,
    language: 'C',
    watchers: 150000,
  },
  {
    project_id: 8514,
    github_url: 'https://github.com/microsoft/vscode',
    score: 0.85,
    language: 'TypeScript',
    watchers: 155000,
  },
  {
    project_id: 6498492,
    github_url: 'https://github.com/pytorch/pytorch',
    score: 0.83,
    language: 'Python',
    watchers: 75000,
  },
  {
    project_id: 41881900,
    github_url: 'https://github.com/vercel/next.js',
    score: 0.81,
    language: 'JavaScript',
    watchers: 115000,
  },
  {
    project_id: 1062897,
    github_url: 'https://github.com/apache/spark',
    score: 0.79,
    language: 'Scala',
    watchers: 38000,
  },
  {
    project_id: 21289110,
    github_url: 'https://github.com/docker/docker',
    score: 0.77,
    language: 'Go',
    watchers: 68000,
  },
  {
    project_id: 45717250,
    github_url: 'https://github.com/fastapi/fastapi',
    score: 0.75,
    language: 'Python',
    watchers: 68000,
  },
  {
    project_id: 54346799,
    github_url: 'https://github.com/rust-lang/rust',
    score: 0.73,
    language: 'Rust',
    watchers: 85000,
  },
  {
    project_id: 2325298,
    github_url: 'https://github.com/golang/go',
    score: 0.71,
    language: 'Go',
    watchers: 120000,
  },
  {
    project_id: 126577260,
    github_url: 'https://github.com/microsoft/TypeScript',
    score: 0.69,
    language: 'TypeScript',
    watchers: 95000,
  },
  {
    project_id: 29028775,
    github_url: 'https://github.com/rails/rails',
    score: 0.67,
    language: 'Ruby',
    watchers: 55000,
  },
  {
    project_id: 9384267,
    github_url: 'https://github.com/nodejs/node',
    score: 0.65,
    language: 'JavaScript',
    watchers: 105000,
  },
]

/**
 * Generate mock recommendations for a user
 * Simulates API delay and returns realistic data
 */
export async function getMockRecommendations(
  userId: number,
  topK: number
): Promise<RecommendationResponse> {
  // Simulate API delay (500ms - 1.5s)
  const delay = Math.random() * 1000 + 500
  await new Promise((resolve) => setTimeout(resolve, delay))

  // Return top K recommendations
  const recommendations = MOCK_RECOMMENDATIONS.slice(0, Math.min(topK, MOCK_RECOMMENDATIONS.length))

  return {
    user_id: userId,
    recommendations: recommendations.map((rec, index) => ({
      ...rec,
      // Slightly vary scores based on rank for realism
      score: rec.score - index * 0.01,
    })),
  }
}

/**
 * Check if mock mode is enabled
 * Can be controlled via environment variable or localStorage
 */
export function isMockModeEnabled(): boolean {
  // Check environment variable first
  if (import.meta.env.VITE_USE_MOCK_DATA === 'true') {
    return true
  }

  // Check localStorage (for runtime toggle)
  const stored = localStorage.getItem('useMockData')
  if (stored !== null) {
    return stored === 'true'
  }

  // Default: use mock data if API URL is localhost (development)
  return false
}

/**
 * Set mock mode
 */
export function setMockMode(enabled: boolean): void {
  localStorage.setItem('useMockData', enabled.toString())
}

