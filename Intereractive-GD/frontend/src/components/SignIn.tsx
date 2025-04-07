import React from "react";
import { useNavigate } from "react-router-dom";
import { GoogleLogin } from '@react-oauth/google';

const SignIn: React.FC = () => {
  const navigate = useNavigate();

  const handleGoogleSignIn = async (credentialResponse: any) => {
    console.log("Google Sign-In response:", credentialResponse.credential);

    try {
      const res = await fetch('http://localhost:8080/api/auth/google', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token: credentialResponse.credential }),
      });

      const data = await res.json();
      console.log("Backend response:", data);
      
      if (data.success) {
        // Store user info in localStorage or state management
        localStorage.setItem('user', JSON.stringify(data.user));
        // Redirect to t"opic page
        navigate("/topic");
      } else {
        console.error("Authentication failed:", data.error);
      }
    } catch (error) {
      console.error("Error during authentication:", error);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center">
      {/* Background Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#4f4f4f2e_1px,transparent_1px)] bg-[size:14px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

      <div className="relative z-10 w-full max-w-md">
        <div className="bg-gray-900/50 p-8 rounded-2xl shadow-xl border border-gray-800">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-2">Welcome Back</h2>
            <p className="text-gray-400">Sign in to continue to Interactive GD</p>
          </div>

          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={handleGoogleSignIn}
              onError={() => {
                console.error("Login Failed");
              }}
              useOneTap
            />
          </div>

          <div className="mt-6 text-center">
            <button
              onClick={() => navigate("/")}
              className="text-yellow-500 hover:text-yellow-400 transition-colors"
            >
              Return to Home
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignIn;
