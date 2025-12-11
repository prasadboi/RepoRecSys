import { useMemo, useState } from 'react'
import './App.css'
import UserView from './components/UserView'
import AdminView from './components/AdminView'
import RecommendationResults from './components/RecommendationResults'
import { fetchStatus, getRecommendations, RecommendationResponse, triggerIngest, triggerReload, triggerTrain } from './services/api'

type ViewMode = 'user' | 'admin'

function App() {
  const [view, setView] = useState<ViewMode>('user')
  const [apiUrl, setApiUrl] = useState<string>(() => import.meta.env.VITE_API_URL || 'http://localhost:8000')
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [adminMessage, setAdminMessage] = useState<string | null>(null)

  const pageTitle = useMemo(() => (view === 'user' ? 'User View' : 'Admin View'), [view])

  const handleRecommend = async (userId: string, userName: string, topK: number) => {
    setLoading(true)
    setError(null)
    setRecommendations(null)
    try {
      const resp = await getRecommendations(userId, topK, apiUrl, userName)
      setRecommendations(resp)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to fetch recommendations'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleAdminAction = async (action: 'ingest' | 'train' | 'reload' | 'status') => {
    setLoading(true)
    setError(null)
    setAdminMessage(null)
    try {
      if (action === 'ingest') {
        const resp = await triggerIngest(apiUrl)
        setAdminMessage(resp.status || 'Ingestion started')
      } else if (action === 'train') {
        const resp = await triggerTrain(apiUrl)
        setAdminMessage(resp.status || 'Training started')
      } else if (action === 'status') {
        const resp = await fetchStatus(apiUrl)
        setAdminMessage(
          `Last: ${resp.last_action || 'n/a'} @ ${resp.last_update || 'n/a'}\n` +
          (resp.logs.slice(-5).join('\n') || 'No logs')
        )
      } else {
        const resp = await triggerReload(apiUrl)
        setAdminMessage(resp.status || 'Reload triggered')
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Admin action failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <p className="eyebrow">GitHub Repository Recommendation System</p>
          <h1>{pageTitle}</h1>
          <p className="subtitle">
            Two-Tower recommender with FastAPI backend. Demo includes user recommendations and admin triggers.
          </p>
        </div>
        <div className="view-toggle">
          <button
            className={view === 'user' ? 'tab active' : 'tab'}
            onClick={() => {
              setError(null)
              setAdminMessage(null)
              setRecommendations(null)
              setView('user')
            }}
          >
            User
          </button>
          <button
            className={view === 'admin' ? 'tab active' : 'tab'}
            onClick={() => {
              setError(null)
              setAdminMessage(null)
              setRecommendations(null)
              setView('admin')
            }}
          >
            Admin
          </button>
        </div>
      </header>

      <main className="main">
        <section className="panel">
          {view === 'user' ? (
            <UserView
              apiUrl={apiUrl}
              onApiUrlChange={setApiUrl}
              onRecommend={handleRecommend}
              loading={loading}
            />
          ) : (
            <AdminView
              apiUrl={apiUrl}
              onApiUrlChange={setApiUrl}
              onAction={handleAdminAction}
              loading={loading}
              message={adminMessage}
            />
          )}
        </section>

        <section className="panel">
          {error && <div className="alert error">{error}</div>}
          {loading && <div className="alert info">Working...</div>}

          {view === 'user' ? (
            recommendations ? (
              <RecommendationResults data={recommendations} />
            ) : (
              !loading && <div className="placeholder">Enter a user and fetch recommendations.</div>
            )
          ) : (
            !loading &&
            !error && (
              <div className="placeholder">
                Use the admin actions to trigger ingest/train/reload for the demo.
              </div>
            )
          )}
        </section>
      </main>
    </div>
  )
}

export default App

