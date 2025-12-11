import { RecommendationResponse } from '../services/api'

interface RecommendationResultsProps {
  recommendations: RecommendationResponse
}

export default function RecommendationResults({ recommendations }: RecommendationResultsProps) {
  const { user_id, recommendations: recs } = recommendations

  if (!recs || recs.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📭</div>
        <p>No recommendations found for User {user_id}</p>
      </div>
    )
  }

  return (
    <div>
      <div className="results-header">
        <h2>Recommendations</h2>
        <div className="results-count">
          {recs.length} recommendation{recs.length !== 1 ? 's' : ''} for User {user_id}
        </div>
      </div>

      <div className="recommendations-grid">
        {recs.map((rec, index) => (
          <div key={rec.project_id} className="repo-card">
            <div className="repo-rank">#{index + 1}</div>
            <div className="repo-title">Project {rec.project_id}</div>
            <a
              href={rec.github_url}
              target="_blank"
              rel="noopener noreferrer"
              className="repo-link"
            >
              {rec.github_url}
            </a>

            <div className="repo-meta">
              <div className="meta-item">
                <span className="meta-label">Language:</span>
                <span>{rec.language || 'N/A'}</span>
              </div>
              <div className="meta-item">
                <span className="meta-label">Watchers:</span>
                <span>{rec.watchers.toLocaleString()}</span>
              </div>
            </div>

            <div className="repo-score">
              <div className="score-label">Recommendation Score</div>
              <div className="score-value">{(rec.score * 100).toFixed(2)}%</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

