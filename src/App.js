// src/App.js
import React from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Sidebar from './Components/Sidebar';
import Home from './Components/Home';
import Dashboard from './Components/Dashboard';
// import SavedLists from './Components/SavedLists';
import SustainabilityTips from './Components/SustainabilityTips';
import LoginPage from './Components/Login';
import SignupPage from './Components/SignUp';
import { CartProvider } from './Components/CartContext';

function App() {
  const location = useLocation();

  // Hide sidebar on login and signup routes
  const hideSidebarPaths = ['/login', '/signup'];
  const showSidebar = !hideSidebarPaths.includes(location.pathname);

  return (
    // ✅ Wrapping everything inside CartProvider so all components have access to cart
    <CartProvider>
      <div className="bg-white-600 min-h-screen flex">
        {/* Sidebar shown conditionally */}
        {showSidebar && <Sidebar />}

        {/* Main content area */}
        <div className="flex-grow bg-emerald-600">
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/" element={<Home />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/tips" element={<SustainabilityTips />} />
          </Routes>
        </div>
      </div>
    </CartProvider>
  );
}

export default App;
