// src/components/LoginPage.jsx
import React, { useState } from "react";
import { FaUserCircle, FaLock } from "react-icons/fa";
import { useNavigate, Link } from "react-router-dom";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "../firebase";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await signInWithEmailAndPassword(auth, email, password);
      alert("Login successful! Redirecting to home.");
      navigate("/");
    } catch (err) {
      setError("Invalid email or password.");
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen w-screen bg-emerald-600 flex items-center justify-center font-sans p-4">
      <div className="bg-white p-8 rounded-xl shadow-xl border border-gray-100 w-full max-w-lg">
        <div className="flex justify-center mb-6">
          <FaUserCircle className="text-emerald-600 text-7xl" />
        </div>
        <h2 className="text-4xl font-extrabold text-center text-emerald-800 mb-6">Welcome Back!</h2>
        <p className="text-lg text-gray-600 text-center mb-8">Sign in to track your green impact.</p>

        <form onSubmit={handleLogin} className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-gray-700 text-md font-semibold mb-2">
              Email or Username
            </label>
            <div className="relative">
              <FaUserCircle className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 text-xl" />
              <input
                type="email"
                id="email"
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-400 focus:border-transparent outline-none text-gray-700"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>

          <div>
            <label htmlFor="password" className="block text-gray-700 text-md font-semibold mb-2">
              Password
            </label>
            <div className="relative">
              <FaLock className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 text-xl" />
              <input
                type="password"
                id="password"
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-400 focus:border-transparent outline-none text-gray-700"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          {error && <p className="text-red-600 text-sm text-center">{error}</p>}

          <button
            type="submit"
            className="w-full bg-gradient-to-br from-emerald-500 to-emerald-600 text-white py-3 rounded-lg font-bold text-lg hover:from-emerald-600 hover:to-emerald-700 transition-all duration-300 shadow-md"
            disabled={loading}
          >
            {loading ? "Logging In..." : "Log In"}
          </button>
        </form>

        <p className="text-center text-gray-600 mt-8 text-md">
          Don't have an account?{" "}
          <Link to="/signup" className="text-emerald-600 hover:underline font-bold">
            Sign Up
          </Link>
        </p>
      </div>
    </div>
  );
}
