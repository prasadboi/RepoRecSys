import { RecommendationResponse } from '../services/api'

interface Props {
  data: RecommendationResponse
}

export default function RecommendationResults({ data }: Props) {
  if (!data.recommendations.length) {
    return <div className="placeholder">No recommendations returned.</div>
  }

  return (
    <div>
      <h3>Top {data.recommendations.length} for user {data.user_id}</h3>
      <div className="results-grid">
        {data.recommendations.map((rec, idx) => (
          <div className="card" key={rec.project_id}>
            <h3>#{idx + 1} &middot; {rec.project_id}</h3>
            <div className="meta">Score: {rec.score.toFixed(4)}</div>
            <div className="meta">Language: {rec.language ?? 'N/A'}</div>
            <div className="meta">Watchers: {rec.watchers ?? 0}</div>
            <a className="link" href={rec.github_url ?? '#'} target="_blank" rel="noreferrer">
              {rec.github_url}
            </a>
            {rec.is_fallback && <div className="alert info" style={{ marginTop: 8 }}>Fallback (global top-K)</div>}
          </div>
        ))}
      </div>
    </div>
  )
}

