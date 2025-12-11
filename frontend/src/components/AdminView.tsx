interface Props {
  apiUrl: string
  onApiUrlChange: (url: string) => void
  onAction: (action: 'ingest' | 'train' | 'reload') => void
  loading: boolean
  message: string | null
}

export default function AdminView({ apiUrl, onApiUrlChange, onAction, loading, message }: Props) {
  return (
    <div className="admin-actions">
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

      <button className="button" type="button" disabled={loading} onClick={() => onAction('ingest')}>
        {loading ? 'Working...' : 'Trigger Ingest'}
      </button>

      <button className="button" type="button" disabled={loading} onClick={() => onAction('train')}>
        {loading ? 'Working...' : 'Trigger Train'}
      </button>

      <button className="button secondary" type="button" disabled={loading} onClick={() => onAction('reload')}>
        {loading ? 'Working...' : 'Reload Model'}
      </button>

      <button className="button secondary" type="button" disabled={loading} onClick={() => onAction('status')}>
        {loading ? 'Working...' : 'Refresh Status'}
      </button>

      {message && <div className="alert info">{message}</div>}
    </div>
  )
}

