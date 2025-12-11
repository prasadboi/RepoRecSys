import { useState } from 'react'

interface Props {
  apiUrl: string
  onApiUrlChange: (url: string) => void
  onRecommend: (userId: string, userName: string, topK: number) => void
  loading: boolean
}

export default function UserView({ apiUrl, onApiUrlChange, onRecommend, loading }: Props) {
  const [userId, setUserId] = useState('1')
  const [userName, setUserName] = useState('')
  const [topK, setTopK] = useState(10)

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    onRecommend(userId.trim(), userName.trim(), topK)
  }

  return (
    <form onSubmit={submit}>
      <div className="form-group">
        <label className="form-label">API URL</label>
        <input
          className="form-input api-url-input"
          value={apiUrl}
          onChange={(e) => onApiUrlChange(e.target.value)}
          placeholder="http://localhost:8000"
          disabled={loading}
        />
      </div>

      <div className="form-group">
        <label className="form-label">User ID</label>
        <input
          className="form-input"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="user id"
          required
          disabled={loading}
        />
      </div>

      <div className="form-group">
        <label className="form-label">GitHub Username (optional, for new users)</label>
        <input
          className="form-input"
          value={userName}
          onChange={(e) => setUserName(e.target.value)}
          placeholder="github username (optional)"
          disabled={loading}
        />
      </div>

      <div className="form-group">
        <label className="form-label">Top K</label>
        <div className="slider-row">
          <input
            type="range"
            min={5}
            max={50}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            disabled={loading}
          />
          <span className="slider-value">{topK}</span>
        </div>
      </div>

      <button type="submit" className="button" disabled={loading}>
        {loading ? 'Loading...' : 'Get Recommendations'}
      </button>
    </form>
  )
}

