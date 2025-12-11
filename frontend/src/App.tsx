import { useState } from 'react'
import './App.css'
import RecommendationForm from './components/RecommendationForm'
import RecommendationResults from './components/RecommendationResults'
import { getRecommendations, RecommendationResponse } from './services/api'
import { isMockModeEnabled, setMockMode } from './services/mockData'

function App() {
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiUrl, setApiUrl] = useState<string>(() => {
    // Default to localhost:8000, but can be overridden via environment variable
    return import.meta.env.VITE_API_URL || 'http://localhost:8000'
  })
  const [useMockData, setUseMockData] = useState<boolean>(() => {
    // TODO: When backend is ready, change this to false
    // Default to true for development (showing mock data)
    // Set to false when you want to connect to real API
    return isMockModeEnabled() || true // ⚠️ CHANGE TO false WHEN BACKEND IS READY
  })

  const handleGetRecommendations = async (userId: number, topK: number) => {
    setLoading(true)
    setError(null)
    setRecommendations(null)

    try {
      const result = await getRecommendations(userId, topK, apiUrl, useMockData)
      setRecommendations(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch recommendations')
    } finally {
      setLoading(false)
    }
  }

  const handleToggleMockData = (enabled: boolean) => {
    setUseMockData(enabled)
    setMockMode(enabled)
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>⭐ GitHub Repository Recommender</h1>
          <p>Get personalized repository recommendations powered by Two-Tower Neural Collaborative Filtering</p>
        </header>

        <div className="main-content">
          <aside className="sidebar">
            <RecommendationForm
              onSubmit={handleGetRecommendations}
              loading={loading}
              apiUrl={apiUrl}
              onApiUrlChange={setApiUrl}
              useMockData={useMockData}
              onToggleMockData={handleToggleMockData}
            />
          </aside>

          <main className="results-section">
            {error && (
              <div className="error">
                <strong>Error:</strong> {error}
              </div>
            )}

            {loading && (
              <div className="loading">
                <div className="spinner"></div>
                <p>Fetching recommendations...</p>
              </div>
            )}

            {!loading && !error && !recommendations && (
              <div className="info">
                <p>👆 Enter a User ID and click "Get Recommendations" to start</p>
                {useMockData && (
                  <p style={{ marginTop: '1rem', fontSize: '0.9rem', opacity: 0.8 }}>
                    💡 Mock data mode is enabled - you'll see sample recommendations
                  </p>
                )}
              </div>
            )}

            {!loading && recommendations && (
              <RecommendationResults recommendations={recommendations} />
            )}
          </main>
        </div>

        <footer className="footer">
          <p><strong>GitHub Repository Recommendation System</strong></p>
          <p>Powered by Two-Tower Neural Collaborative Filtering | Graduate Cloud & ML Course</p>
        </footer>
      </div>
    </div>
  )
}

export default App

