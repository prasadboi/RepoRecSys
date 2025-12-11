import { useState, FormEvent } from 'react'

interface RecommendationFormProps {
  onSubmit: (userId: number, topK: number) => void
  loading: boolean
  apiUrl: string
  onApiUrlChange: (url: string) => void
  useMockData: boolean
  onToggleMockData: (enabled: boolean) => void
}

export default function RecommendationForm({
  onSubmit,
  loading,
  apiUrl,
  onApiUrlChange,
  useMockData,
  onToggleMockData,
}: RecommendationFormProps) {
  const [userId, setUserId] = useState<number>(1)
  const [topK, setTopK] = useState<number>(10)

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (userId > 0 && topK > 0) {
      onSubmit(userId, topK)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <h2>Settings</h2>

      <div className="form-group">
        <label htmlFor="mock-toggle" className="checkbox-label">
          <input
            id="mock-toggle"
            type="checkbox"
            checked={useMockData}
            onChange={(e) => onToggleMockData(e.target.checked)}
            disabled={loading}
            className="checkbox-input"
          />
          <span>Use Mock Data (Demo Mode)</span>
        </label>
        <small className="form-help">
          {useMockData 
            ? 'Showing sample data. Uncheck to connect to real API.' 
            : 'Using real API. Check to see sample data.'}
        </small>
      </div>

      <div className="form-group">
        <label htmlFor="api-url">API URL</label>
        <input
          id="api-url"
          type="text"
          value={apiUrl}
          onChange={(e) => onApiUrlChange(e.target.value)}
          placeholder="http://localhost:8000"
          disabled={loading || useMockData}
        />
        {useMockData && (
          <small className="form-help">API URL disabled in mock mode</small>
        )}
      </div>

      <div className="form-group">
        <label htmlFor="user-id">User ID</label>
        <input
          id="user-id"
          type="number"
          min="1"
          value={userId}
          onChange={(e) => setUserId(parseInt(e.target.value) || 1)}
          disabled={loading}
          required
        />
      </div>

      <div className="slider-group">
        <label htmlFor="top-k">
          Number of Recommendations: <span className="slider-value">{topK}</span>
        </label>
        <div className="slider-container">
          <input
            id="top-k"
            type="range"
            min="5"
            max="50"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            className="slider"
            disabled={loading}
          />
        </div>
      </div>

      <button type="submit" className="btn-primary" disabled={loading}>
        {loading ? 'Loading...' : useMockData ? 'Get Mock Recommendations' : 'Get Recommendations'}
      </button>
    </form>
  )
}

