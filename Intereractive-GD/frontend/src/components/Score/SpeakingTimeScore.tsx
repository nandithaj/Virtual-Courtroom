import { useState, useEffect } from "react";
import { Mic } from "lucide-react";

interface SpeakingTimeScoreProps {
  userId: string;
}

const SpeakingTimeScore: React.FC<SpeakingTimeScoreProps> = ({ userId }) => {
  const [speakingStats, setSpeakingStats] = useState<{
    average_percentage: number;
    total_sessions: number;
  }>({
    average_percentage: 0,
    total_sessions: 0
  });

  useEffect(() => {
    fetchSpeakingStats(userId);
  }, [userId]);

  const fetchSpeakingStats = async (userId: string) => {
    try {
      const response = await fetch(`http://localhost:8080/api/user/speaking-stats/${userId}`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setSpeakingStats({
            average_percentage: data.average_percentage,
            total_sessions: data.total_sessions
          });
        }
      }
    } catch (error) {
      console.error("Error fetching speaking stats:", error);
    }
  };

  // Calculate score out of 10 based on speaking time percentage
  // Ideal range is 20-30%, so:
  // - 20-30% = 10 points
  // - 15-20% or 30-35% = 7 points
  // - 10-15% or 35-40% = 5 points
  // - 5-10% or 40-45% = 3 points
  // - <5% or >45% = 1 point
  const calculateScore = (percentage: number): number => {
    if (percentage >= 20 && percentage <= 30) return 10;
    if ((percentage >= 15 && percentage < 20) || (percentage > 30 && percentage <= 35)) return 7;
    if ((percentage >= 10 && percentage < 15) || (percentage > 35 && percentage <= 40)) return 5;
    if ((percentage >= 5 && percentage < 10) || (percentage > 40 && percentage <= 45)) return 3;
    return 1;
  };

  const score = calculateScore(speakingStats.average_percentage);
  
  // Calculate actual speaking time in minutes and seconds for a 4-minute session
  const calculateSpeakingTime = (percentage: number): string => {
    const totalSeconds = 240; // 4 minutes = 240 seconds
    const speakingSeconds = Math.round((percentage / 100) * totalSeconds);
    const minutes = Math.floor(speakingSeconds / 60);
    const seconds = speakingSeconds % 60;
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  return (
    <div className="mt-6 bg-gray-900/50 border border-gray-800 rounded-lg overflow-hidden">
      <div className="p-4">
        <h3 className="text-lg font-bold">Speaking Time Analysis</h3>
        <p className="text-gray-400 text-sm">
          Based on your {speakingStats.total_sessions} GD sessions
        </p>
      </div>
      <div className="p-4 pt-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <h4 className="text-lg font-medium flex items-center">
              <Mic className="mr-2 h-5 w-5 text-yellow-500" />
              Speaking Time
            </h4>
            <div className="mt-4">
              <div className="text-4xl font-bold text-yellow-500">{speakingStats.average_percentage}%</div>
              <p className="text-sm text-gray-400 mt-1">
                Average speaking time in sessions
              </p>
              <div className="mt-4">
                <div className="flex justify-between text-xs mb-1">
                  <span>Ideal range</span>
                  <span>20-30%</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full relative">
                  <div className="absolute h-full w-[60%] bg-gray-700 rounded-full"></div>
                  <div 
                    className="absolute h-full bg-yellow-500 rounded-full"
                    style={{ width: `${Math.min(speakingStats.average_percentage, 100)}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <h4 className="text-lg font-medium flex items-center">
              <span className="mr-2 h-5 w-5 text-yellow-500">★</span>
              Speaking Score
            </h4>
            <div className="mt-4">
              <div className="text-4xl font-bold text-yellow-500">{score}/10</div>
              <p className="text-sm text-gray-400 mt-1">
                Based on optimal speaking time
              </p>
              <div className="mt-4">
                <p className="text-sm text-gray-300">
                  In a 4-minute session, you speak for approximately:
                </p>
                <p className="text-xl font-bold text-white mt-2">
                  {calculateSpeakingTime(speakingStats.average_percentage)}
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  out of 4:00 total session time
                </p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-gray-800/50 border border-gray-700 rounded-lg">
          <h4 className="text-lg font-medium">Scoring Formula</h4>
          <p className="text-sm text-gray-400 mt-2">
            Your score is calculated based on how close your speaking time is to the ideal range (20-30% of session time).
          </p>
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h5 className="font-medium text-yellow-500">Score Breakdown:</h5>
              <ul className="text-sm text-gray-300 mt-2 space-y-1">
                <li>• 20-30% = 10 points (Optimal)</li>
                <li>• 15-20% or 30-35% = 7 points (Good)</li>
                <li>• 10-15% or 35-40% = 5 points (Fair)</li>
                <li>• 5-10% or 40-45% = 3 points (Needs improvement)</li>
                <li>• &lt;5% or &gt;45% = 1 point (Significant improvement needed)</li>
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-yellow-500">Time Calculation:</h5>
              <p className="text-sm text-gray-300 mt-2">
                For a 4-minute (240 seconds) session:
              </p>
              <ul className="text-sm text-gray-300 mt-2 space-y-1">
                <li>• Optimal: 48-72 seconds</li>
                <li>• Good: 36-48 or 72-84 seconds</li>
                <li>• Fair: 24-36 or 84-96 seconds</li>
                <li>• Needs improvement: 12-24 or 96-108 seconds</li>
                <li>• Significant improvement needed: &lt;12 or &gt;108 seconds</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SpeakingTimeScore; 