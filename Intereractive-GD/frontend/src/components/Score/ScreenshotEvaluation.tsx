import React, { useEffect, useState } from 'react';
import axios from 'axios';

interface AttentionMetrics {
  eyes_closed_count: number;
  head_turned_count: number;
}

interface AttentionScore {
  eyes_closed_ratio: number;
  head_turned_ratio: number;
}

interface FaceDetection {
  total_faces: number;
  frame_status: string;
}

interface ScreenshotSummary {
  total_screenshots: number;
  valid_screenshots: number;
  attention_metrics: AttentionMetrics;
  attention_score: AttentionScore;
}

interface EvaluationData {
  _id: string;
  user_id: string;
  screenshots: Array<{
    analysis?: {
      face_detection: FaceDetection;
      "Left Eye Status": string;
      "Right Eye Status": string;
      "Head Position": string;
    };
    error?: string;
  }>;
  summary: ScreenshotSummary;
}

const LoadingScreen: React.FC = () => {
  return (
    <div className="fixed inset-0 bg-white bg-opacity-90 flex flex-col items-center justify-center z-50">
      <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-blue-500 mb-4"></div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-2">Evaluating your results</h2>
      <p className="text-gray-600">Please wait while we process your data...</p>
    </div>
  );
};

const ScreenshotEvaluation: React.FC<{ userId: string }> = ({ userId }) => {
  const [evaluationData, setEvaluationData] = useState<EvaluationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchEvaluation = async () => {
      try {
        const response = await axios.get(`/api/screenshots/evaluate/${userId}`);
        setEvaluationData(response.data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch evaluation data');
        setLoading(false);
      }
    };

    fetchEvaluation();
  }, [userId]);

  const calculateScores = () => {
    if (!evaluationData) return { eyeScore: 0, faceScore: 0, averageScore: 0 };

    const totalScreenshots = evaluationData.summary.total_screenshots;
    const eyesClosedCount = evaluationData.summary.attention_metrics.eyes_closed_count;
    const headTurnedCount = evaluationData.summary.attention_metrics.head_turned_count;

    // Calculate eye score (100 if eyes open, 0 if closed)
    const eyeScore = ((totalScreenshots - eyesClosedCount) / totalScreenshots) * 100;

    // Calculate face score (100 if straight, 0 if turned)
    const faceScore = ((totalScreenshots - headTurnedCount) / totalScreenshots) * 100;

    // Calculate average score
    const averageScore = (eyeScore + faceScore) / 2;

    return {
      eyeScore: Math.round(eyeScore),
      faceScore: Math.round(faceScore),
      averageScore: Math.round(averageScore)
    };
  };

  if (loading) return <LoadingScreen />;
  
  if (error) return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="bg-red-50 p-4 rounded-lg border border-red-200">
        <p className="text-red-600">{error}</p>
      </div>
    </div>
  );
  
  if (!evaluationData) return (
    <div className="flex items-center justify-center min-h-[400px]">
      <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
        <p className="text-gray-600">No evaluation data available</p>
      </div>
    </div>
  );

  const scores = calculateScores();

  return (
    <div className="p-6 max-w-5xl mx-auto bg-gradient-to-br from-gray-50 to-white rounded-xl shadow-lg">
      <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
        Screenshot Evaluation Results
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow duration-300">
          <h3 className="text-xl font-semibold text-gray-700 mb-4 flex items-center">
            <svg className="w-6 h-6 mr-2 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Screenshot Summary
          </h3>
          <div className="space-y-3">
            <p className="text-gray-600 flex items-center">
              <span className="font-medium">Total Screenshots:</span>
              <span className="ml-2 text-gray-800">{evaluationData.summary.total_screenshots}</span>
            </p>
            <p className="text-gray-600 flex items-center">
              <span className="font-medium">Valid Screenshots:</span>
              <span className="ml-2 text-gray-800">{evaluationData.summary.valid_screenshots}</span>
            </p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow duration-300">
          <h3 className="text-xl font-semibold text-gray-700 mb-4 flex items-center">
            <svg className="w-6 h-6 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Attention Metrics
          </h3>
          <div className="space-y-3">
            <p className="text-gray-600 flex items-center">
              <span className="font-medium">Eyes Closed Count:</span>
              <span className="ml-2 text-gray-800">{evaluationData.summary.attention_metrics.eyes_closed_count}</span>
            </p>
            <p className="text-gray-600 flex items-center">
              <span className="font-medium">Head Turned Count:</span>
              <span className="ml-2 text-gray-800">{evaluationData.summary.attention_metrics.head_turned_count}</span>
            </p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow duration-300">
          <h3 className="text-xl font-semibold text-gray-700 mb-4 flex items-center">
            <svg className="w-6 h-6 mr-2 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
            </svg>
            Scores
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-600 font-medium">Eye Score:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                scores.eyeScore >= 80 ? 'bg-green-100 text-green-800' :
                scores.eyeScore >= 60 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {scores.eyeScore}%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600 font-medium">Face Score:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                scores.faceScore >= 80 ? 'bg-green-100 text-green-800' :
                scores.faceScore >= 60 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {scores.faceScore}%
              </span>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-gray-100">
              <span className="text-gray-800 font-semibold">Average Score:</span>
              <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                scores.averageScore >= 80 ? 'bg-green-100 text-green-800' :
                scores.averageScore >= 60 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {scores.averageScore}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScreenshotEvaluation;
