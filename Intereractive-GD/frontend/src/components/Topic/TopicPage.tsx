import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const topics = [
  "Impact of Artificial Intelligence on Job Markets",
  "Cryptocurrencies: Future of Finance or Bubble?",
  "Role of Social Media in Shaping Public Opinion",
  "The Rise of Electric Vehicles: Opportunities and Challenges",
  "Sustainability in Fashion: Necessity or Trend?",
  "The Influence of ChatGPT on Education and Learning",
  "Work from Home vs. Office: The Future of Work",
  "Data Privacy in the Age of Surveillance Capitalism",
  "Climate Change and Its Impact on Global Economies",
  "Space Exploration: Should We Prioritize Mars Colonization?",
  "Gaming and Mental Health: Boon or Bane?",
  "India's Role in Shaping the Global Economy in 2025",
];

const TopicPage = () => {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const userId = user?.user_id;
  const storedTopic = localStorage.getItem("currentTopic");
  const storedTimeLeft = localStorage.getItem("timeLeft");
  
  const [topic, setTopic] = useState(storedTopic || topics[0]);
  const [timeLeft, setTimeLeft] = useState(storedTimeLeft ? parseInt(storedTimeLeft) : 20);
  const navigate = useNavigate();

  useEffect(() => {
    if (!storedTopic) {
      localStorage.setItem("currentTopic", topic);
    }
    if (!storedTimeLeft) {
      localStorage.setItem("timeLeft", timeLeft.toString());
    }
  }, []);

  useEffect(() => {
    if (timeLeft <= 0) {
      navigate(`/topic/${userId}`);
      return;
    }

    const timer = setInterval(() => {
      setTimeLeft((prevTime) => {
        const newTime = prevTime - 1;
        localStorage.setItem("timeLeft", newTime.toString());
        return newTime;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft, navigate, userId]);

  // Format time as MM:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  const getRandomTopic = () => {
    const currentIndex = topics.indexOf(topic);
    let randomIndex;
    do {
      randomIndex = Math.floor(Math.random() * topics.length);
    } while (randomIndex === currentIndex);
    return topics[randomIndex];
  };

  const handleTopicChange = () => {
    const newTopic = getRandomTopic();
    setTopic(newTopic);
    localStorage.setItem("currentTopic", newTopic);
    // Reset timer to 60 seconds when topic changes
    setTimeLeft(20);
    localStorage.setItem("timeLeft", "20");
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      {/* Background Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

      {/* Header */}
      <header className="relative z-10 border-b border-white/10">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="text-2xl font-bold text-white">
                Interactive GD
              </div>
            </div>
            <div className="flex items-center gap-4">
              <div className="px-4 py-2 bg-yellow-500/10 border border-yellow-500/20 rounded-md">
                <span className="font-mono text-xl font-bold text-yellow-500">
                  {formatTime(timeLeft)}
                </span>
              </div>
              {user?.user_id ? (
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <img
                      src={user.photo_url}
                      alt="User profile"
                      className="w-8 h-8 rounded-full border-2 border-yellow-500"
                    />
                    <span className="text-white">{user.name}</span>
                  </div>
                  <button
                    className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                    onClick={() => {
                      localStorage.removeItem("user");
                      navigate("/signin");
                    }}
                  >
                    Sign out
                  </button>
                </div>
              ) : (
                <>
                  <button
                    className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
                    onClick={() => navigate("/signin")}
                  >
                    Sign in
                  </button>
                  <button
                    className="px-4 py-2 text-sm bg-yellow-500 text-black rounded-md hover:bg-yellow-400 transition-colors"
                    onClick={() => navigate("/get-started")}
                  >
                    Get Started
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Topic Section */}
      <section className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 py-12">
        <div className="max-w-4xl w-full mx-auto text-center space-y-8">
          <div className="inline-flex items-center rounded-full border border-yellow-500/20 bg-yellow-500/10 px-3 py-1 text-sm text-yellow-500 mx-auto">
            <button
              className="hover:bg-yellow-500/20 transition-colors rounded-full px-3 py-1"
              onClick={handleTopicChange}
            >
              Change Topic
            </button>
          </div>

          <div className="p-8 rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-yellow-500 mb-2">
              Your GD Topic
            </h1>
            <div className="text-2xl md:text-3xl lg:text-4xl font-bold mt-4 mb-8">
              {topic}
            </div>

            <div className="mt-12 text-left">
              <h2 className="text-xl font-bold text-yellow-500 mb-4">
                Group Discussion Rules:
              </h2>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">1.</span>
                  <span>
                    Listen actively and respectfully to other participants
                    without interrupting.
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">2.</span>
                  <span>
                    Speak clearly and concisely, staying on topic and within
                    time limits.
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">3.</span>
                  <span>
                    Support your arguments with relevant facts, examples, and
                    logical reasoning.
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">4.</span>
                  <span>
                    Maintain a balanced perspective by considering multiple
                    viewpoints.
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">5.</span>
                  <span>
                    Avoid dominating the discussion; ensure everyone has an
                    opportunity to contribute.
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">6.</span>
                  <span>
                    Be respectful of differing opinions and avoid personal
                    attacks.
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">7.</span>
                  <span>
                    Summarize key points periodically to help maintain focus and
                    clarity.
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="text-yellow-500 mr-2">8.</span>
                  <span>
                    Conclude with a brief summary of the main points discussed
                    and any consensus reached.
                  </span>
                </li>
              </ul>
            </div>
          </div>

          <div className="flex justify-center gap-4 mt-8">
            <button
              className="px-6 py-3 bg-yellow-500 text-black rounded-md hover:bg-yellow-400 transition-colors"
              onClick={() => navigate("/topic/${userId}")}
            >
              Start Discussion
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 py-8">
        <div className="container mx-auto px-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-400">
              © 2025 Interactive GD. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default TopicPage;
