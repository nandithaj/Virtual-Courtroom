import React, { useEffect, useState } from 'react';
import axios from 'axios';

interface GrammarScores {
  final_readability_score: number;
  final_grammar_score: number;
  final_repetitiveness_score: number;
}

interface GrammarEvaluationProps {
  userId: string;
}

const GrammarEvaluation: React.FC<GrammarEvaluationProps> = ({ userId }) => {
  const [scores, setScores] = useState<GrammarScores | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchGrammarScores = async () => {
      try {
        const response = await axios.get(`/api/grammar/${userId}`);
        setScores(response.data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch grammar scores');
        setLoading(false);
      }
    };

    fetchGrammarScores();
  }, [userId]);

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-4 min-h-[200px] flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-4">
        <p className="text-red-500">{error}</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-4">
      <h2 className="text-xl font-semibold mb-4">Grammar Evaluation</h2>
      {scores && (
        <div className="space-y-2">
          <p className="text-gray-700">
            Readability Score: <span className="font-medium">{scores.final_readability_score.toFixed(1)}%</span>
          </p>
          <p className="text-gray-700">
            Grammar Score: <span className="font-medium">{scores.final_grammar_score.toFixed(1)}%</span>
          </p>
          <p className="text-gray-700">
            Repetitiveness Score: <span className="font-medium">{scores.final_repetitiveness_score.toFixed(1)}%</span>
          </p>
        </div>
      )}
    </div>
  );
};

export default GrammarEvaluation; 