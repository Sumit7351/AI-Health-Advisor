import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Select from 'react-select';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

const API_URL = 'http://127.0.0.1:5000';

function App() {
  const [symptoms, setSymptoms] = useState([]);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch symptoms
    axios.get(`${API_URL}/symptoms`)
      .then(res => {
        const options = res.data.symptoms.map(s => ({ value: s, label: s.replace(/_/g, ' ') }));
        setSymptoms(options);
      })
      .catch(err => console.error("Error fetching symptoms", err));
  }, []);

  const handlePredict = () => {
    if (selectedSymptoms.length === 0) {
      alert("Please select at least one symptom.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const symptomList = selectedSymptoms.map(s => s.value);

    axios.post(`${API_URL}/predict`, { symptoms: symptomList })
      .then(res => {
        setResult(res.data);
        setLoading(false);
      })
      .catch(err => {
        setError("Failed to get prediction. Please try again.");
        setLoading(false);
        console.error(err);
      });
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>AI Health Advisor</h1>
        <p>Your personal AI-powered health assistant. Enter your symptoms to get a diagnosis and advice.</p>
      </header>

      <main className="main-content">
        <div className="input-section">
          <h2>Describe your Symptoms</h2>
          <Select
            isMulti
            name="symptoms"
            options={symptoms}
            className="basic-multi-select"
            classNamePrefix="select"
            onChange={setSelectedSymptoms}
            placeholder="Search and select symptoms..."
            styles={{
              control: (base) => ({ ...base, backgroundColor: '#2a2a2a', borderColor: '#444', color: 'white' }),
              menu: (base) => ({ ...base, backgroundColor: '#2a2a2a' }),
              option: (base, state) => ({ ...base, backgroundColor: state.isFocused ? '#444' : '#2a2a2a', color: 'white' }),
              multiValue: (base) => ({ ...base, backgroundColor: '#007bff' }),
              multiValueLabel: (base) => ({ ...base, color: 'white' }),
              multiValueRemove: (base) => ({ ...base, color: 'white', ':hover': { backgroundColor: '#0056b3', color: 'white' } }),
              input: (base) => ({ ...base, color: 'white' }),
              singleValue: (base) => ({ ...base, color: 'white' }),
            }}
          />
          <button className="predict-btn" onClick={handlePredict} disabled={loading}>
            {loading ? 'Analyzing...' : 'Diagnose'}
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="results-section">
            <div className="prediction-card">
              <h2>Predicted Condition</h2>
              <div className="disease-name">{result.prediction.disease}</div>
              <div className="confidence">Confidence: {(result.prediction.confidence * 100).toFixed(1)}%</div>

              <div className="explanation-section">
                <h3>Why this prediction? (AI Explanation)</h3>
                <p>Top contributing factors based on your symptoms:</p>
                <div style={{ width: '100%', height: 300 }}>
                  <ResponsiveContainer>
                    <BarChart
                      data={result.prediction.explanation}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis type="number" stroke="#ccc" />
                      <YAxis dataKey="col_name" type="category" width={150} stroke="#ccc" />
                      <Tooltip contentStyle={{ backgroundColor: '#333', borderColor: '#555' }} />
                      <Legend />
                      <Bar dataKey="feature_importance_vals" name="Importance" fill="#82ca9d" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="advice-section">
              <h3>Health Advice & Recommendations</h3>
              <div className="advice-grid">
                <div className="advice-card">
                  <h4>💊 Cure & Treatment</h4>
                  <ul>
                    {result.info.cure?.map((item, i) => <li key={i}>{item}</li>)}
                  </ul>
                </div>
                <div className="advice-card">
                  <h4>🛡️ Prevention</h4>
                  <ul>
                    {result.info.prevention?.map((item, i) => <li key={i}>{item}</li>)}
                  </ul>
                </div>
                <div className="advice-card">
                  <h4>🥗 Diet</h4>
                  <ul>
                    {result.info.diet?.map((item, i) => <li key={i}>{item}</li>)}
                  </ul>
                </div>
                <div className="advice-card">
                  <h4>🧘 Lifestyle</h4>
                  <ul>
                    {result.info.lifestyle?.map((item, i) => <li key={i}>{item}</li>)}
                  </ul>
                </div>
              </div>
              <div className="disclaimer">
                <strong>Disclaimer:</strong> This is an AI-generated prediction for demonstration purposes only. Please consult a medical professional for accurate diagnosis and treatment.
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
