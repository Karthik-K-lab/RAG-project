import { useState } from 'react'
import './App.css'

function App() {
  const [context, setContext] = useState('')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [topChunks, setTopChunks] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    if (!context.trim() || !question.trim()) {
      setError('Please enter both content and question.')
      return
    }

    setLoading(true)
    setError('')
    setAnswer('')
    setTopChunks([])

    try {
      const response = await fetch('https://rag-project-9md1.onrender.com/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          context: context,
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = await response.json()
      setAnswer(data.answer || 'No answer returned.')
      setTopChunks(data.top_chunks || [])
    } catch (err) {
      setError(err.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="card">
        <h1 className="title">RAG Question Answering</h1>
        <p className="subtitle">Enter content first, then ask a question about it.</p>

        <label className="label">Content</label>
        <textarea
          className="textarea"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          placeholder="Paste or type the content here..."
          rows={8}
        />

        <label className="label">Question</label>
        <input
          className="input"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              handleSubmit()
            }
          }}
          placeholder="Ask your question here..."
        />

        <button className="button" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Thinking...' : 'Submit'}
        </button>

        {loading && (
          <div className="loading-box">
            <div className="spinner"></div>
            <span>Generating answer...</span>
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        {answer && (
          <div className="result-box">
            <h2>Answer</h2>
            <p>{answer}</p>
          </div>
        )}

        {topChunks.length > 0 && (
          <div className="chunks-box">
            <h2>Top Chunks</h2>
            {topChunks.map((chunk, index) => (
              <div key={index} className="chunk-item">
                <strong>{index + 1}.</strong> {chunk}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
